import pytest
import numpy as np
import tempfile
import os
import shutil
from iterative_train import iterative_training
from model import PolicyNet, save_model


class TestIterativeTraining:
    def test_iterative_training_basic(self):
        """Test basic iterative training loop."""
        # Create temporary directory for test data
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save current directory
            old_cwd = os.getcwd()
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)

            try:
                # Change to temp directory
                os.chdir(tmpdir)

                # Create dummy human demos
                states = np.random.randn(100, 9).astype(np.float32)
                actions = np.random.randint(0, 3, 100)
                np.savez('data/human_demos.npz', states=states, actions=actions)

                # Create dummy imitation policy
                policy = PolicyNet(state_dim=9, n_actions=3)
                save_model(policy, 'data/policy_imitation.pt')

                # Run iterative training (small test)
                history = iterative_training(
                    num_iterations=2,
                    mc_episodes=3,
                    mc_K=2,
                    mc_H=5,
                    train_epochs=5,
                    mc_weight=2.0,
                    eval_episodes=5
                )

                # Check history
                assert history is not None
                assert 'iteration' in history
                assert 'mean_score' in history
                assert 'max_score' in history
                assert len(history['iteration']) >= 2  # At least initial + 1 iteration

                # Check files created
                assert os.path.exists('data/policy_mc.pt')
                assert os.path.exists('data/mc_demos.npz')
                assert os.path.exists('data/training_history.npz')

            finally:
                os.chdir(old_cwd)

    def test_iterative_training_no_human_demos(self):
        """Test error handling when no human demos exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)

            try:
                os.chdir(tmpdir)

                # No human demos
                history = iterative_training(num_iterations=1, mc_episodes=1)

                # Should return None
                assert history is None

            finally:
                os.chdir(old_cwd)

    def test_iterative_training_no_imitation_policy(self):
        """Test error handling when no imitation policy exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)

            try:
                os.chdir(tmpdir)

                # Create human demos but no policy
                states = np.random.randn(50, 9).astype(np.float32)
                actions = np.random.randint(0, 3, 50)
                np.savez('data/human_demos.npz', states=states, actions=actions)

                history = iterative_training(num_iterations=1, mc_episodes=1)

                # Should return None
                assert history is None

            finally:
                os.chdir(old_cwd)

    def test_iterative_training_tracks_improvement(self):
        """Test that training tracks improvement over iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)

            try:
                os.chdir(tmpdir)

                # Create dummy data
                states = np.random.randn(100, 9).astype(np.float32)
                actions = np.random.randint(0, 3, 100)
                np.savez('data/human_demos.npz', states=states, actions=actions)

                policy = PolicyNet(state_dim=9, n_actions=3)
                save_model(policy, 'data/policy_imitation.pt')

                # Run training
                history = iterative_training(
                    num_iterations=3,
                    mc_episodes=5,
                    mc_K=2,
                    mc_H=5,
                    train_epochs=5,
                    eval_episodes=10
                )

                # Check history structure
                assert len(history['iteration']) == len(history['mean_score'])
                assert len(history['iteration']) == len(history['max_score'])
                assert len(history['iteration']) >= 3  # Initial + at least 2 iterations

                # Check all scores are valid
                assert all(score >= 0 for score in history['mean_score'])
                assert all(score >= 0 for score in history['max_score'])

            finally:
                os.chdir(old_cwd)
