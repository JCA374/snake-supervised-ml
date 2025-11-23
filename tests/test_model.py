import pytest
import torch
import numpy as np
import tempfile
import os
from model import PolicyNet, save_model, load_model


class TestPolicyNet:
    def test_initialization(self):
        """Test model initializes correctly."""
        model = PolicyNet(state_dim=9, n_actions=3, hidden_dim=64)

        assert isinstance(model, torch.nn.Module)
        assert model.net is not None

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = PolicyNet(state_dim=9, n_actions=3)
        state = torch.randn(1, 9)

        logits = model(state)

        assert logits.shape == (1, 3)

    def test_forward_batch(self):
        """Test forward pass with batch."""
        model = PolicyNet(state_dim=9, n_actions=3)
        batch_size = 32
        states = torch.randn(batch_size, 9)

        logits = model(states)

        assert logits.shape == (batch_size, 3)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        model = PolicyNet(state_dim=9, n_actions=3)
        state = np.random.randn(9).astype(np.float32)

        action = model.get_action(state, deterministic=True)

        assert isinstance(action, int)
        assert 0 <= action <= 2

    def test_get_action_stochastic(self):
        """Test stochastic action selection."""
        model = PolicyNet(state_dim=9, n_actions=3)
        state = np.random.randn(9).astype(np.float32)

        # Sample multiple actions
        actions = [model.get_action(state, deterministic=False) for _ in range(10)]

        assert all(isinstance(a, int) for a in actions)
        assert all(0 <= a <= 2 for a in actions)

    def test_get_action_consistency(self):
        """Test deterministic action is consistent."""
        model = PolicyNet(state_dim=9, n_actions=3)
        state = np.random.randn(9).astype(np.float32)

        action1 = model.get_action(state, deterministic=True)
        action2 = model.get_action(state, deterministic=True)

        assert action1 == action2

    def test_get_action_probs_batch_matches_single(self):
        """Batch probability helper should match single-state queries."""
        model = PolicyNet(state_dim=9, n_actions=3)
        states = np.random.randn(4, 9).astype(np.float32)

        batch_probs = model.get_action_probs_batch(states)
        assert batch_probs.shape == (4, 3)

        for idx, state in enumerate(states):
            single_probs = model.get_probs(state)
            assert np.allclose(batch_probs[idx], single_probs)

    def test_get_probs(self):
        """Test probability output."""
        model = PolicyNet(state_dim=9, n_actions=3)
        state = np.random.randn(9).astype(np.float32)

        probs = model.get_probs(state)

        assert probs.shape == (3,)
        assert np.allclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_get_probs_with_tensor(self):
        """Test probability output with tensor input."""
        model = PolicyNet(state_dim=9, n_actions=3)
        state = torch.randn(9)

        probs = model.get_probs(state)

        assert probs.shape == (3,)
        assert np.allclose(np.sum(probs), 1.0)

    def test_save_and_load(self):
        """Test saving and loading model."""
        model = PolicyNet(state_dim=9, n_actions=3, hidden_dim=64)

        # Get initial prediction
        state = np.random.randn(9).astype(np.float32)
        probs_before = model.get_probs(state)

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            temp_path = f.name

        try:
            save_model(model, temp_path)

            # Load model
            loaded_model = load_model(temp_path, state_dim=9, n_actions=3, hidden_dim=64)

            # Get prediction from loaded model
            probs_after = loaded_model.get_probs(state)

            # Should be identical
            assert np.allclose(probs_before, probs_after)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_gradient_flow(self):
        """Test gradients flow through network."""
        model = PolicyNet(state_dim=9, n_actions=3)
        state = torch.randn(1, 9, requires_grad=True)
        target = torch.tensor([1])  # target action

        logits = model(state)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)

    def test_different_hidden_dims(self):
        """Test model works with different hidden dimensions."""
        for hidden_dim in [32, 64, 128]:
            model = PolicyNet(state_dim=9, n_actions=3, hidden_dim=hidden_dim)
            state = torch.randn(1, 9)

            logits = model(state)
            assert logits.shape == (1, 3)

    def test_eval_mode(self):
        """Test model can switch to eval mode."""
        model = PolicyNet(state_dim=9, n_actions=3)
        model.eval()

        state = np.random.randn(9).astype(np.float32)
        action = model.get_action(state, deterministic=True)

        assert isinstance(action, int)

    def test_output_range(self):
        """Test logits can be any real number."""
        model = PolicyNet(state_dim=9, n_actions=3)
        state = torch.randn(1, 9)

        logits = model(state)

        # Logits can be any real number (not bounded)
        assert torch.all(torch.isfinite(logits))
