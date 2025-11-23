import pytest
import numpy as np
import tempfile
import os
from self_play import generate_self_play_data
from model import PolicyNet, save_model


class TestSelfPlay:
    def test_generate_self_play_data_basic(self):
        """Test basic self-play data generation."""
        # Create a dummy policy
        policy = PolicyNet(state_dim=9, n_actions=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, 'test_policy.pt')
            save_path = os.path.join(tmpdir, 'test_mc_demos.npz')

            # Save policy
            save_model(policy, policy_path)

            # Generate data (small test)
            states, actions = generate_self_play_data(
                policy_path=policy_path,
                save_path=save_path,
                num_episodes=3,
                K=2,
                H=5,
                grid_size=10,
                max_steps=50
            )

            # Check outputs
            assert len(states) > 0
            assert len(actions) > 0
            assert len(states) == len(actions)
            assert states.shape[1] == 9  # state dimension
            assert all(0 <= a <= 2 for a in actions)

            # Check file was saved
            assert os.path.exists(save_path)

            # Check file contents
            data = np.load(save_path)
            assert 'states' in data
            assert 'actions' in data
            assert np.array_equal(data['states'], states)
            assert np.array_equal(data['actions'], actions)

    def test_generate_self_play_data_missing_policy(self):
        """Test error handling when policy doesn't exist."""
        with pytest.raises(FileNotFoundError):
            generate_self_play_data(
                policy_path='nonexistent_policy.pt',
                save_path='test_output.npz',
                num_episodes=1
            )

    def test_generate_self_play_creates_directory(self):
        """Test that output directory is created if it doesn't exist."""
        policy = PolicyNet(state_dim=9, n_actions=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, 'test_policy.pt')
            save_path = os.path.join(tmpdir, 'subdir', 'nested', 'test_mc_demos.npz')

            save_model(policy, policy_path)

            # Directory doesn't exist yet
            assert not os.path.exists(os.path.dirname(save_path))

            # Generate data
            generate_self_play_data(
                policy_path=policy_path,
                save_path=save_path,
                num_episodes=2,
                K=2,
                H=5,
                max_steps=20
            )

            # Directory and file should now exist
            assert os.path.exists(save_path)

    def test_generate_self_play_with_different_params(self):
        """Test self-play with different K and H parameters."""
        policy = PolicyNet(state_dim=9, n_actions=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, 'test_policy.pt')
            save_model(policy, policy_path)

            for K, H in [(1, 5), (3, 10), (5, 20)]:
                save_path = os.path.join(tmpdir, f'mc_K{K}_H{H}.npz')

                states, actions = generate_self_play_data(
                    policy_path=policy_path,
                    save_path=save_path,
                    num_episodes=2,
                    K=K,
                    H=H,
                    max_steps=30
                )

                assert len(states) > 0
                assert len(actions) > 0

    def test_generate_self_play_episodes_produce_data(self):
        """Test that each episode produces some data."""
        policy = PolicyNet(state_dim=9, n_actions=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, 'test_policy.pt')
            save_path = os.path.join(tmpdir, 'test_mc_demos.npz')

            save_model(policy, policy_path)

            states, actions = generate_self_play_data(
                policy_path=policy_path,
                save_path=save_path,
                num_episodes=5,
                K=3,
                H=10,
                grid_size=8,
                max_steps=100
            )

            # Should have generated data from 5 episodes
            assert len(states) >= 5  # at least 1 step per episode
            assert states.shape == (len(states), 9)
            assert actions.shape == (len(actions),)
