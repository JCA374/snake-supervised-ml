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
import numpy as np
from self_play import generate_self_play_data
from train_mc import train_mc
from monte_carlo import evaluate_policy
from model import load_model, PolicyNet
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


def iterative_training(
    num_iterations=10,
    mc_episodes=50,
    mc_K=5,
    mc_H=20,
    train_epochs=100,
    mc_weight=2.0,
    eval_episodes=50,
    record_videos=True,
    video_episodes=10
):
    """
    Run iterative Monte Carlo training loop with comprehensive visualization.

    Args:
        num_iterations: number of improvement iterations
        mc_episodes: episodes to generate per iteration
        mc_K: MC rollouts per action
        mc_H: MC horizon
        train_epochs: training epochs per iteration
        mc_weight: weight for MC data vs human data
        eval_episodes: episodes for evaluation
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

    print(f"\nConfiguration:")
    print(f"  Iterations: {num_iterations}")
    print(f"  MC episodes/iter: {mc_episodes}")
    print(f"  MC rollouts (K): {mc_K}")
    print(f"  MC horizon (H): {mc_H}")
    print(f"  Training epochs: {train_epochs}")
    print(f"  MC weight: {mc_weight}x")
    print(f"  Eval episodes: {eval_episodes}")
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

    # Evaluate initial policy
    print("\n" + "-"*70)
    print("INITIAL EVALUATION (Iteration 0: Imitation)")
    print("-"*70)

    if os.path.exists('data/policy_mc.pt'):
        print("Found existing MC policy, evaluating...")
        initial_policy = load_model('data/policy_mc.pt')
        policy_name = "MC (existing)"
    else:
        print("Evaluating imitation policy...")
        initial_policy = load_model('data/policy_imitation.pt')
        policy_name = "Imitation"

    initial_stats = evaluate_policy(initial_policy, num_episodes=eval_episodes)

    print(f"\n{policy_name} Policy Performance:")
    print(f"  Mean Score: {initial_stats['mean_score']:.2f} ± {initial_stats['std_score']:.2f}")
    print(f"  Max Score:  {initial_stats['max_score']}")
    print(f"  Mean Steps: {initial_stats['mean_length']:.1f}")

    # Save initial model to training logs
    from model import save_model
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
        print(f"\n[1/3] Generating MC self-play data...")
        try:
            states, actions = generate_self_play_data(
                policy_path=None,  # Auto-detect best policy
                save_path='data/mc_demos.npz',
                num_episodes=mc_episodes,
                K=mc_K,
                H=mc_H,
                grid_size=10
            )
            print(f"✓ Generated {len(states)} transitions")
        except Exception as e:
            print(f"❌ Error generating data: {e}")
            break

        # Step 2: Train on combined data
        print(f"\n[2/3] Training on combined data...")
        try:
            model = train_mc(
                human_data_path='data/human_demos.npz',
                mc_data_path='data/mc_demos.npz',
                init_policy_path='data/policy_mc.pt' if os.path.exists('data/policy_mc.pt') else 'data/policy_imitation.pt',
                save_path='data/policy_mc.pt',
                mc_weight=mc_weight,
                epochs=train_epochs,
                batch_size=32,
                lr=5e-4
            )
        except Exception as e:
            print(f"❌ Error training: {e}")
            break

        # Step 3: Evaluate new policy
        print(f"\n[3/4] Evaluating improved policy...")
        try:
            policy = load_model('data/policy_mc.pt')
            stats = evaluate_policy(policy, num_episodes=eval_episodes)

            print(f"\nIteration {iteration} Results:")
            print(f"  Mean Score: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
            print(f"  Max Score:  {stats['max_score']}")
            print(f"  Mean Steps: {stats['mean_length']:.1f}")

            # Calculate improvement
            improvement = stats['mean_score'] - history['mean_score'][-1]
            print(f"  Improvement: {improvement:+.2f}")

            # Update history
            history['iteration'].append(iteration)
            history['mean_score'].append(stats['mean_score'])
            history['std_score'].append(stats['std_score'])
            history['max_score'].append(stats['max_score'])
            history['mean_length'].append(stats['mean_length'])

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
        model_name = "Imitation" if iter_num == 0 else f"MC Iter {iter_num}"
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
