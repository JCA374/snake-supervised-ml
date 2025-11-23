"""
Automated iterative training loop for Snake ML.

This script automates the Monte Carlo self-improvement process:
1. Generate MC self-play data (auto-detects best policy)
2. Train on combined human + MC data
3. Evaluate performance
4. Repeat for N iterations

Run this after you have initial human demos and imitation policy.
"""

import os
import numpy as np
from self_play import generate_self_play_data
from train_mc import train_mc
from monte_carlo import evaluate_policy
from model import load_model


def iterative_training(
    num_iterations=10,
    mc_episodes=50,
    mc_K=5,
    mc_H=20,
    train_epochs=100,
    mc_weight=2.0,
    eval_episodes=50
):
    """
    Run iterative Monte Carlo training loop.

    Args:
        num_iterations: number of improvement iterations
        mc_episodes: episodes to generate per iteration
        mc_K: MC rollouts per action
        mc_H: MC horizon
        train_epochs: training epochs per iteration
        mc_weight: weight for MC data vs human data
        eval_episodes: episodes for evaluation

    Returns:
        history: dict with scores and stats per iteration
    """
    print("\n" + "="*60)
    print("ITERATIVE MONTE CARLO TRAINING")
    print("="*60)

    # Check prerequisites
    if not os.path.exists('data/human_demos.npz'):
        print("\n❌ Error: No human demos found!")
        print("Run: python play_pygame.py")
        return None

    if not os.path.exists('data/policy_imitation.pt'):
        print("\n❌ Error: No imitation policy found!")
        print("Run: python train_imitation.py")
        return None

    print(f"\nConfiguration:")
    print(f"  Iterations: {num_iterations}")
    print(f"  MC episodes/iter: {mc_episodes}")
    print(f"  MC rollouts (K): {mc_K}")
    print(f"  MC horizon (H): {mc_H}")
    print(f"  Training epochs: {train_epochs}")
    print(f"  MC weight: {mc_weight}x")
    print(f"  Eval episodes: {eval_episodes}")

    # Track performance
    history = {
        'iteration': [],
        'mean_score': [],
        'std_score': [],
        'max_score': [],
        'mean_length': []
    }

    # Evaluate initial policy
    print("\n" + "-"*60)
    print("INITIAL EVALUATION")
    print("-"*60)

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
        print(f"\n[3/3] Evaluating improved policy...")
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

        # Check for convergence
        if iteration > 2:
            recent_scores = history['mean_score'][-3:]
            if max(recent_scores) - min(recent_scores) < 0.1:
                print(f"\n⚠ Converged! Scores plateaued (range < 0.1)")
                print(f"Stopping early at iteration {iteration}")
                break

    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    print("\nPerformance over iterations:")
    print(f"{'Iter':<6} {'Mean Score':<12} {'Max Score':<12} {'Improvement':<12}")
    print("-" * 50)

    for i, iter_num in enumerate(history['iteration']):
        improvement = 0 if i == 0 else history['mean_score'][i] - history['mean_score'][i-1]
        print(f"{iter_num:<6} {history['mean_score'][i]:<12.2f} "
              f"{history['max_score'][i]:<12} {improvement:+.2f}")

    best_iter = np.argmax(history['mean_score'])
    print(f"\n✓ Best performance at iteration {history['iteration'][best_iter]}")
    print(f"  Mean Score: {history['mean_score'][best_iter]:.2f}")
    print(f"  Max Score:  {history['max_score'][best_iter]}")

    total_improvement = history['mean_score'][-1] - history['mean_score'][0]
    print(f"\n✓ Total improvement: {total_improvement:+.2f}")

    # Save history
    if len(history['iteration']) > 0:
        np.savez('data/training_history.npz', **history)
        print("\n✓ Training history saved to data/training_history.npz")

    # Watch final agent
    print("\n" + "-"*60)
    print("To watch the final trained agent:")
    print("  python watch_agent.py data/policy_mc.pt")
    print("-"*60)

    return history


if __name__ == '__main__':
    # Run with default settings
    history = iterative_training(
        num_iterations=10,
        mc_episodes=50,
        mc_K=5,
        mc_H=20,
        train_epochs=100,
        mc_weight=2.0,
        eval_episodes=50
    )
