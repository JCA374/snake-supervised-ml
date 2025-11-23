import os
import numpy as np
from model import PolicyNet, save_model
from self_play import generate_self_play_data


def test_generate_self_play_parallel(tmp_path):
    """Parallel self-play generation should create datasets with correct shapes."""
    policy = PolicyNet(state_dim=9, n_actions=3)
    policy_path = tmp_path / "policy.pt"
    save_model(policy, str(policy_path))

    save_path = tmp_path / "mc_parallel.npz"
    states, actions = generate_self_play_data(
        policy_path=str(policy_path),
        save_path=str(save_path),
        num_episodes=4,
        K=1,
        H=5,
        grid_size=6,
        max_steps=50,
        num_workers=2
    )

    assert os.path.exists(save_path)
    assert len(states) == len(actions)
    assert len(states) > 0

    with np.load(save_path) as data:
        assert data['states'].shape[0] == len(states)
        assert data['actions'].shape[0] == len(actions)
