import torch
import torch.nn as nn
import numpy as np

class PolicyNet(nn.Module):
    """
    Policy network for Snake agent.

    Input: 9-dim state feature vector
    Output: 3 action logits (left, straight, right)
    """

    def __init__(self, state_dim=9, n_actions=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            logits: tensor of shape (batch_size, n_actions) or (n_actions,)
        """
        return self.net(x)

    def _states_to_tensor(self, states):
        """
        Convert input states to a 2D float tensor.
        """
        tensor = torch.as_tensor(states, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def get_action_probs_batch(self, states):
        """
        Get action probabilities for a batch of states.
        """
        with torch.no_grad():
            state_tensor = self._states_to_tensor(states)
            logits = self.forward(state_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def get_action(self, state, deterministic=False):
        """
        Get action from state.

        Args:
            state: numpy array of shape (state_dim,)
            deterministic: if True, return argmax action; else sample from policy

        Returns:
            action: int in {0, 1, 2}
        """
        probs = self.get_action_probs_batch(state)[0]

        if deterministic:
            return int(np.argmax(probs))

        return int(np.random.choice(len(probs), p=probs))

    def get_probs(self, state):
        """
        Get action probabilities from state.

        Args:
            state: numpy array of shape (state_dim,) or tensor

        Returns:
            probs: numpy array of shape (n_actions,)
        """
        return self.get_action_probs_batch(state)[0]


def save_model(model, path):
    """Save model to disk."""
    torch.save(model.state_dict(), path)


def load_model(path, state_dim=9, n_actions=3, hidden_dim=64):
    """Load model from disk."""
    model = PolicyNet(state_dim, n_actions, hidden_dim)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
