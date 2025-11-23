# Snake Supervised ML with Monte Carlo Self-Play

Train a neural network to play Snake using **supervised learning** (imitation) followed by **Monte Carlo self-improvement**.

## Overview

This project implements a two-stage learning approach:

1. **Imitation Learning**: Train a policy network to mimic human gameplay
2. **Monte Carlo Self-Play**: Improve the policy by simulating future moves and learning from better decisions

**Key Features:**
- CPU-only (no GPU required)
- Simple 9-dimensional state representation
- Small neural network (64 hidden units)
- Comprehensive test coverage (39 tests)
- Visual gameplay with Pygame

## Project Structure

```
snake-supervised-ml/
├── env.py                  # Snake environment with gym-like interface
├── model.py                # Policy network (PyTorch)
├── play_pygame.py          # Record human gameplay
├── train_imitation.py      # Train on human demos
├── monte_carlo.py          # MC action selection and evaluation
├── self_play.py            # Generate MC improved data
├── train_mc.py             # Train on combined human + MC data
├── watch_agent.py          # Watch trained agent play
├── tests/                  # Test suite
│   ├── test_env.py
│   ├── test_model.py
│   └── test_monte_carlo.py
├── data/                   # Generated data and models
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- numpy
- torch (CPU version)
- pygame
- pytest

## Quick Start

### 1. Record Your Gameplay

Play Snake and record your moves (arrow keys to move):

```bash
python play_pygame.py
```

This saves demonstrations to `data/human_demos.npz`.

### 2. Train Imitation Policy

Train the network to copy your gameplay:

```bash
python train_imitation.py
```

This creates `data/policy_imitation.pt`.

### 3. Watch the Agent

See how well it learned from you:

```bash
python watch_agent.py data/policy_imitation.pt
```

### 4. Generate Monte Carlo Data

Use the current policy with MC rollouts to find better moves:

```bash
python self_play.py
```

This creates `data/mc_demos.npz`.

### 5. Train on Combined Data

Improve the policy by learning from both human demos and MC-improved moves:

```bash
python train_mc.py
```

This creates `data/policy_mc.pt`.

### 6. Compare Performance

Watch the improved agent:

```bash
python watch_agent.py data/policy_mc.pt
```

### 7. Iterate

Repeat steps 4-6 to continue improving:

```bash
# Update self_play.py to use the MC policy
python self_play.py  # generates new MC data
python train_mc.py   # trains on combined data
```

## How It Works

### State Representation (9 dimensions)

1. **Direction** (4 dims): One-hot encoding [up, right, down, left]
2. **Food position** (2 dims): Normalized (dx, dy) relative to head
3. **Danger detection** (3 dims): Binary flags for collision if moving [left, straight, right]

### Actions (3 relative movements)

- `0`: Turn left
- `1`: Go straight
- `2`: Turn right

### Rewards

- `+10.0`: Eating food
- `-10.0`: Collision (wall or self)
- `-0.01`: Each step (encourages efficiency)

### Monte Carlo Action Selection

For each possible action:
1. Clone the environment
2. Take that action
3. Simulate `K` rollouts of `H` steps using the current policy
4. Calculate average return
5. Choose action with highest average return

Parameters:
- `K=5`: Number of rollouts per action
- `H=20`: Horizon (max steps per rollout)

### Training Details

**Imitation Learning:**
- Loss: Cross-entropy
- Optimizer: Adam (lr=1e-3)
- Epochs: 100
- Batch size: 32

**Monte Carlo Training:**
- Combines human demos + MC demos
- MC data weighted 2x higher
- Lower learning rate (lr=5e-4)
- Fine-tunes from imitation policy

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run specific test files:

```bash
python -m pytest tests/test_env.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_monte_carlo.py -v
```

## Customization

### Adjust Grid Size

```python
# In any script
env = SnakeEnv(grid_size=15)  # default is 10
```

### Monte Carlo Parameters

```python
# In self_play.py
generate_self_play_data(
    K=10,   # more rollouts (slower, smarter)
    H=30,   # longer horizon (slower, more foresight)
)
```

### Network Architecture

```python
# In model.py
model = PolicyNet(
    state_dim=9,
    n_actions=3,
    hidden_dim=128  # increase for more capacity
)
```

### Training Hyperparameters

```python
# In train_imitation.py or train_mc.py
train_imitation(
    epochs=200,      # more training
    batch_size=64,   # larger batches
    lr=5e-4,         # adjust learning rate
)
```

## Implementation Notes

### CPU-Only Design

- Small network (64 hidden units)
- Efficient state representation (9 dims)
- Fast environment (numpy arrays)
- No GPU required

### Environment Features

- Gym-like interface (`reset()`, `step()`, `render()`)
- `clone()` method for MC rollouts (using `deepcopy`)
- Relative actions (independent of absolute direction)
- Deterministic with seed

### Key Implementation Details

1. **Arrow keys → Relative actions**: `play_pygame.py` converts absolute arrow keys to relative actions based on current snake direction
2. **Avoid 180° reversal**: Moving backward is illegal, handled by treating as "straight"
3. **MC doesn't modify env**: All rollouts use `env.clone()` to preserve original state
4. **Data weighting**: MC demos are repeated 2x when combining datasets

## Performance Tips

1. **Record diverse gameplay**: Try different strategies when recording
2. **Adjust MC weight**: Higher = more MC influence, lower = more human influence
3. **Iterate**: Run multiple rounds of self-play → train
4. **Monitor validation**: Stop if validation accuracy plateaus

## Troubleshooting

**"No module named pygame"**: Run `pip install pygame`

**"No human demos found"**: Run `python play_pygame.py` first

**Agent performs poorly**:
- Record more/better human demos
- Increase MC rollouts (K parameter)
- Train for more epochs

**Too slow**:
- Reduce K (rollouts per action)
- Reduce H (horizon)
- Use smaller grid size

## Future Enhancements

- Add value network (actor-critic)
- Use full grid observation (CNN)
- Implement MCTS instead of simple MC rollouts
- Add curriculum learning (start with small grids)
- Multi-step returns with discount factor

## License

MIT

## Acknowledgments

Inspired by AlphaZero's policy iteration, simplified for CPU-only Snake gameplay.
