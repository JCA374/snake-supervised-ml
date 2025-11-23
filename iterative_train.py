"""
Automated iterative training loop for Snake ML with comprehensive visualization.

This script automates the Monte Carlo self-improvement process:
1. Evaluate human baseline
2. Generate MC self-play data (auto-detects best policy)
3. Train on combined human + MC data
4. Evaluate performance and record best episode video
5. Save iteration-specific models
6. Repeat for N iterations
7. Generate training visualizations and report

Run this after you have initial human demos and imitation policy.
"""

import os
import shutil
import time
import numpy as np
from self_play import generate_self_play_data
from train_mc import train_mc
from monte_carlo import evaluate_policy
from model import load_model, save_model
from visualize_training import (plot_training_progress, plot_comparison,
                                create_training_report)

# Optional: video recording (requires pygame + imageio)
try:
    from video_recorder import record_best_episode
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    print("Warning: video recording not available (install pygame and imageio)")


def evaluate_human_baseline(human_data_path='data/human_demos.npz',
                           num_episodes_sample=100):
    """
    Estimate human baseline performance from recorded demos.

    Args:
        human_data_path: path to human demonstrations
        num_episodes_sample: number of sample episodes to estimate from

    Returns:
        estimated average score per episode
    """
    if not os.path.exists(human_data_path):
        return None

    data = np.load(human_data_path)
    states = data['states']

    # Estimate episodes by looking for resets (initial states)
    # This is a rough estimate - actual score tracking would be better
    # For now, return a rough estimate based on data volume
    estimated_score = len(states) / (num_episodes_sample * 50)  # Rough heuristic

    print(f"Human demo data: {len(states)} transitions")
    print(f"Estimated human baseline: ~{estimated_score:.1f} score per episode")

    return estimated_score


def _history_best_model_candidate():
    """Return (path, label) for the best historical model if available."""
    history_files = [
        'training_logs/training_history.npz',
        'data/training_history.npz'
    ]

    for history_path in history_files:
        if not os.path.exists(history_path):
            continue

        try:
            with np.load(history_path) as history:
                if 'iteration' not in history or 'mean_score' not in history:
                    continue
                iterations = history['iteration']
                mean_scores = history['mean_score']
                if len(iterations) == 0:
                    continue
                best_idx = int(np.argmax(mean_scores))
                best_iter = int(iterations[best_idx])
        except Exception as exc:
            print(f"⚠ Could not read training history at {history_path}: {exc}")
            continue

        if best_iter == 0:
            candidate = 'training_logs/models/policy_iter_0_imitation.pt'
            label = 'History best (imitation)'
        else:
            candidate = f'training_logs/models/policy_iter_{best_iter}_mc.pt'
            label = f'History best (iteration {best_iter})'

        if os.path.exists(candidate):
            return candidate, label

    return None, None


def select_best_starting_policy(eval_episodes):
    """Evaluate available checkpoints and return the best-performing policy."""
    candidates = []
    history_path, history_label = _history_best_model_candidate()
    if history_path:
        candidates.append((history_path, history_label))

    if os.path.exists('data/policy_mc.pt'):
        candidates.append(('data/policy_mc.pt', 'Current MC policy'))

    if os.path.exists('data/policy_imitation.pt'):
        candidates.append(('data/policy_imitation.pt', 'Imitation policy'))

    # Deduplicate while preserving order
    seen_paths = set()
    unique_candidates = []
    for path, label in candidates:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_candidates.append((path, label))

    if not unique_candidates:
        raise FileNotFoundError("No available policy checkpoints found.")

    print("\nScanning available policies for best starting point...")
    best_result = None
    for path, label in unique_candidates:
        try:
            policy = load_model(path)
            stats = evaluate_policy(policy, num_episodes=eval_episodes)
            print(f"  Candidate {label}: mean score {stats['mean_score']:.2f} "
                  f"(max {stats['max_score']}) from {path}")
        except Exception as exc:
            print(f"  ⚠ Skipping {path}: {exc}")
            continue

        if best_result is None or stats['mean_score'] > best_result['stats']['mean_score']:
            best_result = {
                'policy': policy,
                'stats': stats,
                'path': path,
                'label': label
            }

    if best_result is None:
        raise RuntimeError("Could not load any valid policy checkpoints.")

    return best_result


def iterative_training(
    num_iterations=50,
    mc_episodes=50,
    mc_K=5,
    mc_H=20,
    mc_K_step=1,
    mc_H_step=5,
    mc_adapt_threshold=0.1,
    mc_adapt_patience=2,
    mc_reuse_iters=3,
    mc_num_workers=1,
    train_epochs=100,
    mc_weight=2.0,
    eval_episodes=50,
    fast_eval_episodes=10,
    eval_full_every=5,
    record_videos=True,
    video_episodes=10
):
    """
    Run iterative Monte Carlo training loop with comprehensive visualization.

    Args:
        num_iterations: number of improvement iterations
        mc_episodes: episodes to generate per iteration
        mc_K: MC rollouts per action
        mc_H: MC horizon (starting value)
        mc_K_step: increment applied to K when adaptation triggers
        mc_H_step: increment applied to H when adaptation triggers
        mc_adapt_threshold: improvement threshold that contributes to MC budget increase
        mc_adapt_patience: number of consecutive low-improvement iterations before adapting MC budgets
        mc_reuse_iters: number of past MC datasets to retain for training (None for unlimited)
        mc_num_workers: number of processes for parallel self-play generation
        train_epochs: training epochs per iteration
        mc_weight: weight for MC data vs human data
        eval_episodes: episodes for full evaluation
        fast_eval_episodes: episodes for lightweight evaluation each iteration
        eval_full_every: frequency (iterations) to force full evaluation
        record_videos: whether to record best episode videos
        video_episodes: number of episodes to find best one for video

    Returns:
        history: dict with scores and stats per iteration
    """
    print("\n" + "="*70)
    print("ITERATIVE MONTE CARLO TRAINING WITH VISUALIZATION")
    print("="*70)

    # Check prerequisites
    if not os.path.exists('data/human_demos.npz'):
        print("\n❌ Error: No human demos found!")
        print("Run: python play_pygame.py")
        return None

    if not os.path.exists('data/policy_imitation.pt'):
        print("\n❌ Error: No imitation policy found!")
        print("Run: python train_imitation.py")
        return None

    # Create training logs directory
    os.makedirs('training_logs', exist_ok=True)
    os.makedirs('training_logs/models', exist_ok=True)
    os.makedirs('training_logs/videos', exist_ok=True)
    os.makedirs('training_logs/mc_demos', exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Iterations: {num_iterations}")
    print(f"  MC episodes/iter: {mc_episodes}")
    print(f"  MC rollouts (K): {mc_K} (step +{mc_K_step})")
    print(f"  MC horizon (H): {mc_H} (step +{mc_H_step})")
    print(f"  MC adapt patience: {mc_adapt_patience} iteration(s)")
    print(f"  MC reuse: last {mc_reuse_iters} iterations" if mc_reuse_iters else "  MC reuse: unlimited")
    print(f"  MC workers: {mc_num_workers}")
    print(f"  Training epochs: {train_epochs}")
    print(f"  MC weight: {mc_weight}x")
    print(f"  Eval episodes: {eval_episodes} (fast eval {fast_eval_episodes}, full every {eval_full_every})")
    print(f"  Record videos: {record_videos and VIDEO_AVAILABLE}")

    # Evaluate human baseline
    print("\n" + "-"*70)
    print("HUMAN BASELINE EVALUATION")
    print("-"*70)
    human_baseline = evaluate_human_baseline('data/human_demos.npz')

    # Track performance
    history = {
        'iteration': [],
        'mean_score': [],
        'std_score': [],
        'max_score': [],
        'mean_length': []
    }

    # Track video paths
    video_paths = {}
    mc_demo_paths = []
    current_mc_K = mc_K
    current_mc_H = mc_H
    human_data_in_mix = True
    low_improve_streak = 0

    # Evaluate initial policy
    print("\n" + "-"*70)
    print("INITIAL EVALUATION (Iteration 0)")
    print("-"*70)

    best_start = select_best_starting_policy(eval_episodes)
    initial_policy = best_start['policy']
    initial_stats = best_start['stats']
    policy_name = best_start['label']

    # Ensure canonical MC checkpoint matches the best policy
    os.makedirs('data', exist_ok=True)
    save_model(initial_policy, 'data/policy_mc.pt')

    print(f"\nStarting from {policy_name}:")
    print(f"  Mean Score: {initial_stats['mean_score']:.2f} ± {initial_stats['std_score']:.2f}")
    print(f"  Max Score:  {initial_stats['max_score']}")
    print(f"  Mean Steps: {initial_stats['mean_length']:.1f}")

    if human_baseline is not None and initial_stats['mean_score'] >= human_baseline:
        human_data_in_mix = False
        print("  Human baseline beaten already; will focus on MC data going forward.")

    # Save initial model to training logs
    save_model(initial_policy, 'training_logs/models/policy_iter_0_imitation.pt')
    print(f"✓ Model saved to training_logs/models/policy_iter_0_imitation.pt")

    # Record initial video
    if record_videos and VIDEO_AVAILABLE:
        print(f"\nRecording best episode video (sampling {video_episodes} episodes)...")
        try:
            video_result = record_best_episode(
                initial_policy,
                num_episodes=video_episodes,
                save_path='training_logs/videos/iter_0_imitation',
                grid_size=10,
                fps=10
            )
            if video_result and video_result['video_path']:
                video_paths[0] = video_result['video_path']
                print(f"✓ Video recorded: {video_result['video_path']}")
        except Exception as e:
            print(f"⚠ Video recording failed: {e}")

    history['iteration'].append(0)
    history['mean_score'].append(initial_stats['mean_score'])
    history['std_score'].append(initial_stats['std_score'])
    history['max_score'].append(initial_stats['max_score'])
    history['mean_length'].append(initial_stats['mean_length'])

    # Iterative improvement loop
    for iteration in range(1, num_iterations + 1):
        print("\n" + "="*60)
        print(f"ITERATION {iteration}/{num_iterations}")
        print("="*60)

        # Step 1: Generate MC self-play data
        print(f"\n[1/3] Generating MC self-play data (K={current_mc_K}, H={current_mc_H})...")
        gen_start = time.perf_counter()
        try:
            iteration_mc_path = f'training_logs/mc_demos/iter_{iteration}.npz'
            states, actions = generate_self_play_data(
                policy_path=None,  # Auto-detect best policy
                save_path=iteration_mc_path,
                num_episodes=mc_episodes,
                K=current_mc_K,
                H=current_mc_H,
                grid_size=10,
                num_workers=mc_num_workers
            )
            print(f"✓ Generated {len(states)} transitions in {time.perf_counter() - gen_start:.1f}s")

            # Track MC dataset for reuse
            mc_demo_paths.append(iteration_mc_path)
            if mc_reuse_iters and len(mc_demo_paths) > mc_reuse_iters:
                oldest = mc_demo_paths.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)

            # Aggregate recent MC datasets
            aggregate_states = []
            aggregate_actions = []
            for path in mc_demo_paths:
                with np.load(path) as data:
                    aggregate_states.append(data['states'])
                    aggregate_actions.append(data['actions'])

            combined_states = np.concatenate(aggregate_states, axis=0)
            combined_actions = np.concatenate(aggregate_actions, axis=0)
            aggregate_path = 'data/mc_demos.npz'
            np.savez(aggregate_path, states=combined_states, actions=combined_actions)
            print(f"✓ Aggregated {len(combined_states)} MC transitions "
                  f"from {len(mc_demo_paths)} iteration(s)")

        except Exception as e:
            print(f"❌ Error generating data: {e}")
            break

        # Step 2: Train on combined data
        print(f"\n[2/3] Training on combined data...")
        current_policy_path = 'data/policy_mc.pt' if os.path.exists('data/policy_mc.pt') else 'data/policy_imitation.pt'
        backup_path = os.path.join('training_logs', 'models', '_prev_policy_backup.pt')
        try:
            shutil.copy(current_policy_path, backup_path)
        except Exception as e:
            print(f"❌ Error backing up current policy: {e}")
            break

        if not human_data_in_mix:
            print("  Human data disabled (policy outperforms baseline). Using MC-only training.")

        train_start = time.perf_counter()
        try:
            model = train_mc(
                human_data_path='data/human_demos.npz',
                mc_data_path='data/mc_demos.npz',
                init_policy_path=current_policy_path,
                save_path='data/policy_mc.pt',
                mc_weight=mc_weight,
                epochs=train_epochs,
                batch_size=32,
                lr=5e-4,
                include_human_data=human_data_in_mix
            )
            print(f"✓ Training completed in {time.perf_counter() - train_start:.1f}s")
        except Exception as e:
            print(f"❌ Error training: {e}")
            break

        # Step 3: Evaluate new policy
        print(f"\n[3/4] Evaluating improved policy...")
        eval_start = time.perf_counter()
        try:
            policy = load_model('data/policy_mc.pt')
            # Lightweight evaluation every iteration
            stats_fast = evaluate_policy(policy, num_episodes=fast_eval_episodes)

            run_full_eval = (
                iteration % eval_full_every == 0 or
                history['mean_score'][-1] is None or
                stats_fast['mean_score'] - history['mean_score'][-1] >= mc_adapt_threshold
            )

            if run_full_eval:
                stats = evaluate_policy(policy, num_episodes=eval_episodes)
                print(f"  ✓ Full evaluation run ({time.perf_counter() - eval_start:.1f}s)")
            else:
                stats = stats_fast
                print(f"  ✓ Using fast evaluation stats ({time.perf_counter() - eval_start:.1f}s)")

            print(f"\nIteration {iteration} Results:")
            print(f"  Mean Score: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
            print(f"  Max Score:  {stats['max_score']}")
            print(f"  Mean Steps: {stats['mean_length']:.1f}")

            # Calculate improvement
            improvement = stats['mean_score'] - history['mean_score'][-1]
            print(f"  Improvement: {improvement:+.2f}")

            if improvement < 0:
                print("  Regression detected. Reverting to previous policy for next iteration.")
                shutil.copy(backup_path, 'data/policy_mc.pt')
                policy = load_model('data/policy_mc.pt')
                stats = {
                    'mean_score': history['mean_score'][-1],
                    'std_score': history['std_score'][-1],
                    'max_score': history['max_score'][-1],
                    'mean_length': history['mean_length'][-1]
                }
                print(f"  Restored mean score: {stats['mean_score']:.2f}")
            elif (human_baseline is not None and human_data_in_mix and
                  stats['mean_score'] >= human_baseline):
                human_data_in_mix = False
                print("  Human baseline surpassed. Future iterations will focus on MC data only.")

            # Update history with the policy we will keep
            history['iteration'].append(iteration)
            history['mean_score'].append(stats['mean_score'])
            history['std_score'].append(stats['std_score'])
            history['max_score'].append(stats['max_score'])
            history['mean_length'].append(stats['mean_length'])

            if os.path.exists(backup_path):
                os.remove(backup_path)
            # Adapt MC parameters if we're not seeing sufficient improvement
            if improvement <= mc_adapt_threshold:
                low_improve_streak += 1
                if low_improve_streak >= mc_adapt_patience and iteration < num_iterations:
                    current_mc_K += mc_K_step
                    current_mc_H += mc_H_step
                    low_improve_streak = 0
                    print(f"  Adaptive MC: increasing K to {current_mc_K}, H to {current_mc_H}")
                else:
                    remaining = mc_adapt_patience - low_improve_streak
                    print(f"  Low improvement streak: {low_improve_streak}/{mc_adapt_patience} "
                          f"(need {remaining} more iteration(s) to adapt)")
            else:
                low_improve_streak = 0

        except Exception as e:
            print(f"❌ Error evaluating: {e}")
            break

        # Step 4: Save model and record video
        print(f"\n[4/4] Saving model and recording video...")

        # Save iteration-specific model
        iter_model_path = f'training_logs/models/policy_iter_{iteration}_mc.pt'
        save_model(policy, iter_model_path)
        print(f"✓ Model saved to {iter_model_path}")

        # Record video
        if record_videos and VIDEO_AVAILABLE:
            print(f"Recording best episode video (sampling {video_episodes} episodes)...")
            try:
                video_result = record_best_episode(
                    policy,
                    num_episodes=video_episodes,
                    save_path=f'training_logs/videos/iter_{iteration}_mc',
                    grid_size=10,
                    fps=10
                )
                if video_result and video_result['video_path']:
                    video_paths[iteration] = video_result['video_path']
                    print(f"✓ Video recorded: {video_result['video_path']}")
            except Exception as e:
                print(f"⚠ Video recording failed: {e}")

        # Check for convergence
        if iteration > 2:
            recent_scores = history['mean_score'][-3:]
            if max(recent_scores) - min(recent_scores) < 0.1:
                print(f"\n⚠ Converged! Scores plateaued (range < 0.1)")
                print(f"Stopping early at iteration {iteration}")
                break

    # Final summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    print("\nPerformance over iterations:")
    print(f"{'Iter':<6} {'Model':<15} {'Mean Score':<15} {'Max Score':<12} {'Improvement':<12}")
    print("-" * 70)

    for i, iter_num in enumerate(history['iteration']):
        model_name = policy_name if iter_num == 0 else f"MC Iter {iter_num}"
        improvement = 0 if i == 0 else history['mean_score'][i] - history['mean_score'][i-1]
        mean_score_str = f"{history['mean_score'][i]:.2f} ± {history['std_score'][i]:.2f}"
        print(f"{iter_num:<6} {model_name:<15} {mean_score_str:<15} "
              f"{history['max_score'][i]:<12} {improvement:+.2f}")

    best_iter = np.argmax(history['mean_score'])
    print(f"\n✓ Best performance at iteration {history['iteration'][best_iter]}")
    print(f"  Mean Score: {history['mean_score'][best_iter]:.2f}")
    print(f"  Max Score:  {history['max_score'][best_iter]}")

    total_improvement = history['mean_score'][-1] - history['mean_score'][0]
    print(f"\n✓ Total improvement: {total_improvement:+.2f}")

    if human_baseline:
        vs_human = history['mean_score'][-1] - human_baseline
        print(f"✓ Final score vs human baseline: {vs_human:+.2f}")

    # Save history
    if len(history['iteration']) > 0:
        np.savez('data/training_history.npz', **history)
        np.savez('training_logs/training_history.npz', **history)
        print("\n✓ Training history saved to training_logs/training_history.npz")

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    try:
        # Plot training progress
        plot_training_progress(
            history,
            save_path='training_logs/training_progress.png',
            human_baseline=human_baseline
        )

        # Plot model comparison
        plot_comparison(
            history,
            save_path='training_logs/model_comparison.png',
            human_baseline=human_baseline
        )

        # Create text report
        create_training_report(
            history,
            human_baseline=human_baseline,
            videos=video_paths,
            save_path='training_logs/training_report.txt'
        )

        print("\n✓ All visualizations generated!")
        print("  - training_logs/training_progress.png")
        print("  - training_logs/model_comparison.png")
        print("  - training_logs/training_report.txt")

    except Exception as e:
        print(f"⚠ Visualization generation failed: {e}")

    # Final instructions
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  📊 Plots: training_logs/training_progress.png")
    print("  📊 Comparison: training_logs/model_comparison.png")
    print("  📝 Report: training_logs/training_report.txt")
    print("  🎮 Models: training_logs/models/")
    if video_paths:
        print(f"  🎬 Videos: training_logs/videos/ ({len(video_paths)} videos)")

    print("\nTo watch the final trained agent:")
    print("  python watch_agent.py data/policy_mc.pt")
    print("\nTo watch a specific iteration:")
    print("  python watch_agent.py training_logs/models/policy_iter_N_mc.pt")
    print("="*70)

    return history


if __name__ == '__main__':
    # Run with default settings
    # Adjust parameters as needed:
    # - num_iterations: More iterations = more improvement opportunities
    # - mc_episodes: More episodes = better MC data quality
    # - mc_K: More rollouts = smarter action selection
    # - mc_H: Longer horizon = more foresight
    # - record_videos: Set to False to skip video recording
    # - video_episodes: More episodes = better chance of finding best gameplay

    history = iterative_training(
        num_iterations=10,
        mc_episodes=50,
        mc_K=5,
        mc_H=20,
        train_epochs=100,
        mc_weight=2.0,
        eval_episodes=50,
        record_videos=True,
        video_episodes=10
    )
