# Repository Guidelines

## Project Structure & Module Organization
- Core gameplay and learning logic sits in `env.py`, `model.py`, and `monte_carlo.py`, while `train_imitation.py`, `self_play.py`, `train_mc.py`, and `iterative_train.py` wire those pieces into workflows.
- Capture/visualization scripts (`play_pygame.py`, `watch_agent.py`, `visualize_training.py`, `video_recorder.py`) live at the repo root; every artifact they emit belongs in `data/`, `training_logs/`, or `archive/`.
- Tests live in `tests/` (env/model/monte_carlo); keep fixtures minimal, rely on deterministic 10×10 grids, and document any new helper near the module it exercises.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` – install runtime plus visualization extras.
- `python play_pygame.py` – record demonstrations into `data/human_demos.npz`.
- `python train_imitation.py` and, when tuning, `python watch_agent.py data/policy_imitation.pt` – train and preview the supervised policy.
- `python self_play.py` then `python train_mc.py` – create `data/mc_demos.npz` and fine-tune into `data/policy_mc.pt`.
- `python -m pytest tests/ -v` or `bash run_tests.sh` – run regression tests after any env/model/training edit.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, snake_case for functions/variables, and PascalCase for classes like `SnakeEnv` and `PolicyNet`.
- Add type hints/docstrings when public APIs change and keep helpers side-effect free so Monte Carlo cloning remains deterministic.
- Name artifacts `data/policy_<stage>.pt` or `data/*_demos.npz`; describe unusual hyperparameters near the call site or in README/USAGE snippets.

## Testing Guidelines
- Pytest follows the `tests/test_<module>.py` and `def test_<behavior>()` pattern; every new feature or bug fix deserves a deterministic case.
- Seed RNGs, keep grids small, and favor lightweight fixtures for fast feedback; use focused commands like `python -m pytest tests/test_env.py -k collision` while iterating.
- Before each PR, run the full suite plus any touched script (e.g., `python self_play.py --episodes 10`) and record the results in the PR body.

## Commit & Pull Request Guidelines
- Commits stay short and imperative as in history (`Add training visualization system`, `req improve`); keep subjects under ~50 characters and add optional bodies for context.
- Reference issues/feature IDs, include asset or doc edits with the code they rely on, and avoid mixing unrelated refactors.
- PRs need context, change summary, commands/tests executed, and evidence for visual tweaks (screenshot or GIF from `visualize_training.py`); highlight known risks or follow-ups.

## Data & Configuration Tips
- Store generated demos, checkpoints, and media under `data/` or `training_logs/` and commit only lightweight samples.
- Stick to documented hyperparameters unless README/USAGE explains the experiment so others can reproduce quickly.
