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

    def get_action(self, state, deterministic=False):
        """
        Get action from state.

        Args:
            state: numpy array of shape (state_dim,)
            deterministic: if True, return argmax action; else sample from policy

        Returns:
            action: int in {0, 1, 2}
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits = self.forward(state_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()

            return action

    def get_probs(self, state):
        """
        Get action probabilities from state.

        Args:
            state: numpy array of shape (state_dim,) or tensor

        Returns:
            probs: numpy array of shape (n_actions,)
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            elif len(state.shape) == 1:
                state = state.unsqueeze(0)

            logits = self.forward(state)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            return probs.numpy()


def save_model(model, path):
    """Save model to disk."""
    torch.save(model.state_dict(), path)


def load_model(path, state_dim=9, n_actions=3, hidden_dim=64):
    """Load model from disk."""
    model = PolicyNet(state_dim, n_actions, hidden_dim)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
