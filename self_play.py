import numpy as np
import os
import multiprocessing as mp
from env import SnakeEnv
from model import load_model
from monte_carlo import monte_carlo_action


def _run_self_play(policy, num_episodes, K, H, grid_size, max_steps, progress=False):
    """Generate self-play data sequentially with optional progress logging."""
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
            action, _ = monte_carlo_action(env, policy, K=K, H=H)
            episode_states.append(state.copy())
            episode_actions.append(action)
            state, reward, done, info = env.step(action)
            steps += 1

        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
        episode_scores.append(env.score)

        if progress and (episode + 1) % 10 == 0:
            mean_score = np.mean(episode_scores[-10:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Score: {env.score} | "
                  f"Steps: {steps} | "
                  f"Avg Score (last 10): {mean_score:.2f}")

    return (
        np.array(all_states),
        np.array(all_actions),
        np.array(episode_scores)
    )


def _self_play_worker(args):
    """Worker entry point for multiprocessing self-play generation."""
    (
        policy_path,
        num_episodes,
        K,
        H,
        grid_size,
        max_steps,
        seed
    ) = args

    if seed is not None:
        np.random.seed(seed)

    policy = load_model(policy_path)
    return _run_self_play(
        policy,
        num_episodes,
        K,
        H,
        grid_size,
        max_steps,
        progress=False
    )


def generate_self_play_data(
    policy_path=None,
    save_path='data/mc_demos.npz',
    num_episodes=50,
    K=5,
    H=20,
    grid_size=10,
    max_steps=1000,
    num_workers=1
):
    """
    Generate self-play data using Monte Carlo action selection.

    Args:
        policy_path: path to trained policy (if None, auto-detects best available)
        save_path: path to save MC demonstrations
        num_episodes: number of episodes to generate
        K: number of MC rollouts per action
        H: horizon for MC rollouts
        grid_size: size of game grid
        max_steps: max steps per episode
        num_workers: number of parallel processes (1 = sequential)

    Returns:
        states, actions: numpy arrays of generated data
    """
    # Auto-detect best policy if not specified
    if policy_path is None:
        if os.path.exists('data/policy_mc.pt'):
            policy_path = 'data/policy_mc.pt'
            print("Auto-detected: Using MC-improved policy")
        elif os.path.exists('data/policy_imitation.pt'):
            policy_path = 'data/policy_imitation.pt'
            print("Auto-detected: Using imitation policy")
        else:
            raise FileNotFoundError(
                "No trained policy found. Run train_imitation.py first."
            )

    # Load policy
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    print(f"\n=== Self-Play Data Generation ===")
    print(f"Loading policy from {policy_path}")
    policy = load_model(policy_path)

    print(f"Generating {num_episodes} episodes with MC (K={K}, H={H}, workers={num_workers})")

    if num_workers <= 1:
        states, actions, episode_scores = _run_self_play(
            policy,
            num_episodes,
            K,
            H,
            grid_size,
            max_steps,
            progress=True
        )
    else:
        tasks = []
        base = num_episodes // num_workers
        remainder = num_episodes % num_workers
        for worker_idx in range(num_workers):
            episodes_for_worker = base + (1 if worker_idx < remainder else 0)
            if episodes_for_worker <= 0:
                continue
            tasks.append((
                policy_path,
                episodes_for_worker,
                K,
                H,
                grid_size,
                max_steps,
                np.random.randint(0, 1_000_000)
            ))

        results = []
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_self_play_worker, tasks)

        state_chunks = [res[0] for res in results if len(res[0]) > 0]
        action_chunks = [res[1] for res in results if len(res[1]) > 0]
        score_chunks = [res[2] for res in results if len(res[2]) > 0]

        state_dim = policy.net[0].in_features if hasattr(policy, 'net') else 0
        n_actions = policy.net[-1].out_features if hasattr(policy, 'net') else 0

        states = (np.concatenate(state_chunks, axis=0)
                  if state_chunks else np.empty((0, state_dim)))
        actions = (np.concatenate(action_chunks, axis=0)
                   if action_chunks else np.empty((0,), dtype=np.int64))
        episode_scores = np.concatenate(score_chunks, axis=0) if score_chunks else np.empty((0,))

    # Convert to arrays
    states = np.array(states)
    actions = np.array(actions)

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
        policy_path=None,  # Auto-detect best policy
        save_path='data/mc_demos.npz',
        num_episodes=50,
        K=5,
        H=20,
        grid_size=10
    )
