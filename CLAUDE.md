# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

This project implements a **two-stage learning system** for Snake:

### Stage 1: Supervised Learning (Imitation)
Human plays → Record demos → Train policy to mimic human

### Stage 2: Monte Carlo Self-Improvement
Current policy → MC rollouts to find better actions → Retrain on combined (human + MC) data

### Critical Design Patterns

**1. Relative Action System**
- Actions are RELATIVE, not absolute: `0=turn left`, `1=straight`, `2=turn right`
- `play_pygame.py` converts arrow keys (absolute) → relative actions based on current snake direction
- This makes the policy direction-agnostic (same network output works regardless of which way snake faces)

**2. Environment Cloning for MC Rollouts**
- `env.clone()` uses `copy.deepcopy()` to create independent copies
- MC action selection simulates K rollouts per action WITHOUT modifying the original environment
- Each rollout: clone env → take action → simulate H steps with policy → calculate return
- Critical: rollouts must not affect the actual game state

**3. State Representation (9-dim feature vector)**
- Direction one-hot (4): [up, right, down, left]
- Food relative position (2): normalized (dx, dy)
- Danger detection (3): binary [danger_if_turn_left, danger_if_straight, danger_if_turn_right]
- NOT using full grid observation (keeps network small for CPU)

**4. Data Flow**
```
Human gameplay → human_demos.npz
                      ↓
              policy_imitation.pt (trained on human demos)
                      ↓
              MC self-play (policy simulates futures)
                      ↓
                  mc_demos.npz
                      ↓
      Combined training (human + MC weighted 2x)
                      ↓
              policy_mc.pt (improved policy)
                      ↓
              Iterate: use policy_mc for next MC generation
```

**5. Monte Carlo Action Selection**
For each of 3 possible actions:
- Clone environment K times (default K=5)
- Take the candidate action, then rollout H steps (default H=20) following current policy
- Calculate average return across K rollouts
- Choose action with highest average return
- This provides "lookahead" without explicit search tree

## Commands

### Testing
```bash
# Run all tests (39 tests total)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_env.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_monte_carlo.py -v

# Run single test
python -m pytest tests/test_env.py::TestSnakeEnv::test_clone -v

# Run with short traceback
python -m pytest tests/ -v --tb=short
```

### Training Workflow

```bash
# 1. Record human gameplay (arrow keys to play)
python play_pygame.py

# 2. Train imitation policy
python train_imitation.py

# 3. Watch imitation agent
python watch_agent.py data/policy_imitation.pt

# 4. Generate MC-improved data
python self_play.py

# 5. Train on combined data
python train_mc.py

# 6. Watch improved agent
python watch_agent.py data/policy_mc.pt

# 7. Iterate: update self_play.py to use policy_mc.pt, then repeat 4-6
```

### Evaluation
```bash
# Evaluate policy (no GUI)
python monte_carlo.py  # if policy_imitation.pt exists

# Compare two policies
# Edit monte_carlo.py to load both and call compare_policies()
```

## Key Implementation Details

### Environment (env.py)
- `step(action)` returns `(state, reward, done, info)`
- `clone()` creates deep copy for MC rollouts (critical for MC correctness)
- `_get_state()` computes 9-dim feature vector
- `_is_collision(relative_action)` checks if action would cause death (used for danger detection)
- Rewards: `+10` food, `-10` death, `-0.01` per step

### Model (model.py)
- `PolicyNet`: 2-layer MLP (9 → 64 → 64 → 3)
- `get_action(state, deterministic)`: returns action int (0-2)
- `get_probs(state)`: returns probability distribution (used by MC rollouts)
- Save/load uses PyTorch state dict

### Monte Carlo (monte_carlo.py)
- `monte_carlo_action(env, policy, K, H)`: returns best action and all action values
- `evaluate_policy(policy, num_episodes)`: returns stats dict (mean_score, max_score, etc.)
- `compare_policies(policy1, policy2)`: evaluates and compares two policies

### Training Scripts
- `train_imitation.py`: Standard supervised learning (cross-entropy on human demos)
- `train_mc.py`: Loads both datasets, repeats MC samples by `mc_weight` multiplier, trains on combined data
- Both use 80/20 train/val split and save best model by validation accuracy

### Pygame Scripts
- `play_pygame.py`: Arrow keys → relative actions via `_arrow_to_relative_action()`
- Illegal 180° reverses are ignored (treated as straight)
- Records `(state, action)` pairs during gameplay
- `watch_agent.py`: Similar rendering, but agent chooses actions

## Modifying Hyperparameters

### MC Rollout Parameters
In `self_play.py`:
- `K=5`: More rollouts = smarter but slower (K=10 for better quality)
- `H=20`: Longer horizon = more foresight but slower (H=30 for longer planning)

### Training Parameters
In `train_imitation.py` or `train_mc.py`:
- `epochs=100`: Increase for more training
- `lr=1e-3`: Learning rate (use 5e-4 for fine-tuning in MC training)
- `batch_size=32`: Can increase to 64 for faster training

### MC Data Weighting
In `train_mc.py`:
- `mc_weight=2.0`: Controls MC data influence (higher = more MC, lower = more human)
- Implemented as `mc_repeat = int(round(mc_weight))` to repeat MC samples

### Environment
In any script:
- `grid_size=10`: Increase for harder game (e.g., 15 or 20)
- `max_steps=1000`: Prevents infinite loops

## Common Pitfalls

1. **Don't modify env during MC rollouts** - Always use `env.clone()` before MC simulations
2. **Relative vs absolute actions** - Remember actions are relative to current direction
3. **Arrow key conversion** - Pygame arrow keys must be converted via `_arrow_to_relative_action()`
4. **Data files** - Ensure `data/` directory exists before running scripts
5. **Policy paths** - Update `self_play.py` to use latest policy when iterating (policy_mc.pt instead of policy_imitation.pt)

## Testing New Features

When adding features:
- Environment changes: Add tests to `tests/test_env.py` (especially test `clone()` if modifying state)
- Model changes: Add tests to `tests/test_model.py`
- MC changes: Add tests to `tests/test_monte_carlo.py`
- Always verify MC doesn't modify original env: `test_monte_carlo_does_not_modify_env` pattern
