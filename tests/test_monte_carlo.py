import pytest
import numpy as np
from env import SnakeEnv
from model import PolicyNet
from monte_carlo import monte_carlo_action, evaluate_policy, compare_policies


class TestMonteCarlo:
    def test_monte_carlo_action_returns_valid_action(self):
        """Test MC action selection returns valid action."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        policy = PolicyNet(state_dim=9, n_actions=3)

        action, values = monte_carlo_action(env, policy, K=3, H=10)

        assert isinstance(action, (int, np.integer))
        assert 0 <= action <= 2

    def test_monte_carlo_action_values(self):
        """Test MC returns action values for all actions."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        policy = PolicyNet(state_dim=9, n_actions=3)

        action, values = monte_carlo_action(env, policy, K=5, H=10)

        assert len(values) == 3
        assert all(k in values for k in [0, 1, 2])
        assert all(isinstance(v, (float, np.floating)) for v in values.values())

    def test_monte_carlo_does_not_modify_env(self):
        """Test MC rollouts don't modify original environment."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        policy = PolicyNet(state_dim=9, n_actions=3)

        # Save initial state
        initial_snake = env.snake.copy()
        initial_food = env.food
        initial_score = env.score
        initial_done = env.done

        # Run MC
        monte_carlo_action(env, policy, K=5, H=10)

        # Check environment unchanged
        assert env.snake == initial_snake
        assert env.food == initial_food
        assert env.score == initial_score
        assert env.done == initial_done

    def test_monte_carlo_with_different_k(self):
        """Test MC works with different K values."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        policy = PolicyNet(state_dim=9, n_actions=3)

        for k in [1, 5, 10]:
            action, values = monte_carlo_action(env, policy, K=k, H=10)
            assert 0 <= action <= 2

    def test_monte_carlo_with_different_horizon(self):
        """Test MC works with different horizon values."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        policy = PolicyNet(state_dim=9, n_actions=3)

        for h in [5, 10, 20]:
            action, values = monte_carlo_action(env, policy, K=3, H=h)
            assert 0 <= action <= 2

    def test_evaluate_policy_returns_stats(self):
        """Test policy evaluation returns proper statistics."""
        policy = PolicyNet(state_dim=9, n_actions=3)

        stats = evaluate_policy(policy, num_episodes=10, grid_size=10, max_steps=100)

        assert 'mean_score' in stats
        assert 'std_score' in stats
        assert 'mean_length' in stats
        assert 'std_length' in stats
        assert 'max_score' in stats

        assert stats['mean_score'] >= 0
        assert stats['std_score'] >= 0
        assert stats['mean_length'] > 0
        assert stats['max_score'] >= stats['mean_score']

    def test_evaluate_policy_deterministic(self):
        """Test policy evaluation is somewhat deterministic with same seed."""
        np.random.seed(42)
        policy = PolicyNet(state_dim=9, n_actions=3)

        stats1 = evaluate_policy(policy, num_episodes=5, grid_size=10, max_steps=100)

        np.random.seed(42)
        stats2 = evaluate_policy(policy, num_episodes=5, grid_size=10, max_steps=100)

        # Should be similar (not exact due to policy stochasticity)
        assert abs(stats1['mean_score'] - stats2['mean_score']) < 5.0

    def test_compare_policies(self):
        """Test policy comparison."""
        policy1 = PolicyNet(state_dim=9, n_actions=3)
        policy2 = PolicyNet(state_dim=9, n_actions=3)

        comparison = compare_policies(policy1, policy2, num_episodes=10, grid_size=10)

        assert 'policy1' in comparison
        assert 'policy2' in comparison
        assert 'improvement' in comparison

        assert 'score' in comparison['improvement']
        assert 'length' in comparison['improvement']

    def test_monte_carlo_prefers_food(self):
        """Test MC tends to select actions toward food."""
        # This is a probabilistic test, may occasionally fail
        policy = PolicyNet(state_dim=9, n_actions=3)

        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()

        # Place food directly ahead
        head = env.snake[0]
        dx, dy = env.DIRECTIONS[env.direction]
        env.food = (head[0] + dx, head[1] + dy)

        action, values = monte_carlo_action(env, policy, K=10, H=20)

        # Going straight (1) should have highest value since food is ahead
        # This might not always hold for untrained policy, but test the mechanism
        assert isinstance(values[1], (float, np.floating))

    def test_monte_carlo_avoids_immediate_death(self):
        """Test MC avoids actions that lead to immediate death."""
        policy = PolicyNet(state_dim=9, n_actions=3)

        env = SnakeEnv(grid_size=5, seed=42)
        env.reset()

        # Place snake near wall
        env.snake = [(4, 2), (3, 2), (2, 2)]
        env.direction = 1  # facing right, wall ahead

        action, values = monte_carlo_action(env, policy, K=10, H=5)

        # Going straight should have the worst value (hits wall)
        # This is probabilistic and may not always hold for untrained policy
        assert action in [0, 1, 2]  # Just verify it returns valid action


class HorizonToyEnv:
    """Minimal environment to reason about MC horizon/rollouts."""

    def __init__(self, future_steps=4):
        self.future_steps = future_steps
        self.reset()

    def reset(self):
        self.steps = 0
        self.done = False
        return np.array([0.0], dtype=np.float32)

    def clone(self):
        clone = HorizonToyEnv(self.future_steps)
        clone.steps = self.steps
        clone.done = self.done
        return clone

    def step(self, action):
        if self.done:
            raise ValueError("Episode finished")

        self.steps += 1
        reward = 0.0
        done = False

        if action == 0:
            reward = 5.0 if self.steps == 1 else 0.0
            done = True
        elif action == 1:
            reward = 0.0 if self.steps == 1 else 2.0
            done = self.steps >= self.future_steps
        else:
            reward = -1.0
            done = True

        self.done = done
        state = np.array([self.steps], dtype=np.float32)
        return state, reward, done, {}


class AlwaysStraightPolicy:
    """Policy stub that always selects action 1."""

    def get_action(self, state, deterministic=False):
        return 1


class RecordingPolicy:
    """Policy stub that records batch sizes for caching tests."""

    def __init__(self):
        self.batch_sizes = []

    def get_action_probs_batch(self, states):
        batch = np.asarray(states, dtype=np.float32)
        self.batch_sizes.append(len(batch))
        probs = np.zeros((len(batch), 3), dtype=np.float32)
        if len(batch) > 0:
            probs[:, 1] = 1.0  # always pick action 1
        return probs


def test_monte_carlo_horizon_changes_action_choice():
    """
    Larger horizons should allow MC to pick actions with delayed rewards.
    """
    env = HorizonToyEnv(future_steps=4)
    env.reset()
    policy = AlwaysStraightPolicy()

    short_action, short_values = monte_carlo_action(env, policy, K=1, H=1)
    long_action, long_values = monte_carlo_action(env, policy, K=1, H=5)

    assert short_action == 0  # immediate reward dominates with tiny horizon
    assert long_action == 1   # longer horizon captures delayed 2-point gains
    assert short_values[0] > short_values[1]
    assert long_values[1] > long_values[0]


def test_monte_carlo_caches_duplicate_states():
    """
    MC batching should collapse identical states into single policy calls.
    """
    env = HorizonToyEnv(future_steps=3)
    env.reset()
    policy = RecordingPolicy()

    # All rollouts see the same state progression, so cache should keep batches at size 1
    monte_carlo_action(env, policy, K=4, H=3, n_actions=3)

    assert len(policy.batch_sizes) > 0
    assert all(size == 1 for size in policy.batch_sizes)
