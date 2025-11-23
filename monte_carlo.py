import torch
import numpy as np
from env import SnakeEnv
from model import PolicyNet


def monte_carlo_action(env, policy, K=5, H=20, n_actions=3):
    """
    Select action using Monte Carlo rollouts.

    For each possible action, simulate K rollouts of up to H steps,
    following the current policy. Choose the action with the highest
    average return.

    Args:
        env: current environment state (will be cloned)
        policy: PolicyNet model
        K: number of rollouts per action
        H: horizon (max steps per rollout)
        n_actions: number of possible actions

    Returns:
        best_action: int in {0, 1, 2}
        action_values: dict mapping action -> average return
    """
    action_returns = {}

    for action in range(n_actions):
        returns = []

        for _ in range(K):
            # Clone environment for this rollout
            env_copy = env.clone()

            # Take the candidate action
            state, reward, done, _ = env_copy.step(action)
            total_return = reward
            steps = 0

            # Rollout using policy
            while not done and steps < H:
                # Get action from policy
                action_rollout = policy.get_action(state, deterministic=False)

                # Take step
                state, reward, done, _ = env_copy.step(action_rollout)
                total_return += reward
                steps += 1

            returns.append(total_return)

        # Average return for this action
        action_returns[action] = np.mean(returns)

    # Choose best action
    best_action = max(action_returns, key=action_returns.get)

    return best_action, action_returns


def evaluate_policy(policy, num_episodes=100, grid_size=10, max_steps=1000):
    """
    Evaluate policy performance.

    Args:
        policy: PolicyNet model
        num_episodes: number of episodes to run
        grid_size: size of game grid
        max_steps: max steps per episode

    Returns:
        stats: dict with mean/std of scores and episode lengths
    """
    scores = []
    lengths = []

    for _ in range(num_episodes):
        env = SnakeEnv(grid_size=grid_size)
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = policy.get_action(state, deterministic=False)
            state, reward, done, info = env.step(action)
            steps += 1

        scores.append(env.score)
        lengths.append(steps)

    stats = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'max_score': np.max(scores)
    }

    return stats


def compare_policies(policy1, policy2, num_episodes=100, grid_size=10):
    """
    Compare two policies.

    Args:
        policy1: first PolicyNet model
        policy2: second PolicyNet model
        num_episodes: number of episodes for evaluation
        grid_size: size of game grid

    Returns:
        comparison: dict with stats for both policies
    """
    print("\nEvaluating policy 1...")
    stats1 = evaluate_policy(policy1, num_episodes, grid_size)

    print("Evaluating policy 2...")
    stats2 = evaluate_policy(policy2, num_episodes, grid_size)

    comparison = {
        'policy1': stats1,
        'policy2': stats2,
        'improvement': {
            'score': stats2['mean_score'] - stats1['mean_score'],
            'length': stats2['mean_length'] - stats1['mean_length']
        }
    }

    print(f"\nPolicy 1: Score {stats1['mean_score']:.2f} ± {stats1['std_score']:.2f}, "
          f"Length {stats1['mean_length']:.1f}")
    print(f"Policy 2: Score {stats2['mean_score']:.2f} ± {stats2['std_score']:.2f}, "
          f"Length {stats2['mean_length']:.1f}")
    print(f"Improvement: {comparison['improvement']['score']:.2f} score, "
          f"{comparison['improvement']['length']:.1f} length")

    return comparison


if __name__ == '__main__':
    # Example: test Monte Carlo action selection
    from model import load_model
    import os

    if os.path.exists('data/policy_imitation.pt'):
        print("Loading trained policy...")
        policy = load_model('data/policy_imitation.pt')

        print("\nTesting Monte Carlo action selection...")
        env = SnakeEnv(grid_size=10)
        state = env.reset()

        print(f"Initial state: {state}")

        # Get MC action
        best_action, action_values = monte_carlo_action(env, policy, K=10, H=20)

        print(f"Action values: {action_values}")
        print(f"Best action: {best_action}")

        # Evaluate policy
        print("\nEvaluating policy...")
        stats = evaluate_policy(policy, num_episodes=50)
        print(f"Mean score: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
        print(f"Max score: {stats['max_score']}")

    else:
        print("No trained policy found. Run train_imitation.py first.")
