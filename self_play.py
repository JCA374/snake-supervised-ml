import numpy as np
import os
from env import SnakeEnv
from model import load_model
from monte_carlo import monte_carlo_action


def generate_self_play_data(
    policy_path='data/policy_imitation.pt',
    save_path='data/mc_demos.npz',
    num_episodes=50,
    K=5,
    H=20,
    grid_size=10,
    max_steps=1000
):
    """
    Generate self-play data using Monte Carlo action selection.

    Args:
        policy_path: path to trained policy
        save_path: path to save MC demonstrations
        num_episodes: number of episodes to generate
        K: number of MC rollouts per action
        H: horizon for MC rollouts
        grid_size: size of game grid
        max_steps: max steps per episode

    Returns:
        states, actions: numpy arrays of generated data
    """
    # Load policy
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    print(f"\n=== Self-Play Data Generation ===")
    print(f"Loading policy from {policy_path}")
    policy = load_model(policy_path)

    print(f"Generating {num_episodes} episodes with MC (K={K}, H={H})")

    all_states = []
    all_actions = []
    episode_scores = []

    for episode in range(num_episodes):
        env = SnakeEnv(grid_size=grid_size)
        state = env.reset()

        episode_states = []
        episode_actions = []
        steps = 0

        while not env.done and steps < max_steps:
            # Get MC action
            action, _ = monte_carlo_action(env, policy, K=K, H=H)

            # Record
            episode_states.append(state.copy())
            episode_actions.append(action)

            # Take action
            state, reward, done, info = env.step(action)
            steps += 1

        # Store episode data
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
        episode_scores.append(env.score)

        if (episode + 1) % 10 == 0:
            mean_score = np.mean(episode_scores[-10:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Score: {env.score} | "
                  f"Steps: {steps} | "
                  f"Avg Score (last 10): {mean_score:.2f}")

    # Convert to arrays
    states = np.array(all_states)
    actions = np.array(all_actions)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, states=states, actions=actions)

    print(f"\n✓ Generated {len(states)} transitions from {num_episodes} episodes")
    print(f"✓ Mean score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"✓ Max score: {np.max(episode_scores)}")
    print(f"✓ Saved to {save_path}")

    return states, actions


if __name__ == '__main__':
    states, actions = generate_self_play_data(
        policy_path='data/policy_imitation.pt',
        save_path='data/mc_demos.npz',
        num_episodes=50,
        K=5,
        H=20,
        grid_size=10
    )
