# Iterative Monte Carlo Training Overview

This document narrates the end-to-end training process used in `iterative_train.py` so you can follow it linearly with a screen reader or text-to-speech. Each numbered section corresponds to a major phase of the loop and explains the rationale and key parameters engineers might tweak.

## 1. Bootstrapping the Best Available Policy
1. The script first verifies prerequisites: `data/human_demos.npz` must exist alongside the imitation checkpoint `data/policy_imitation.pt`.
2. It scans historical checkpoints (existing `data/policy_mc.pt` plus models saved in `training_logs/models/`) and evaluates each candidate with `evaluate_policy`.
3. The highest scoring model becomes the canonical starting point: it is copied into `data/policy_mc.pt`, logged as iteration zero, and used for both self-play and training warm starts.

## 2. Baseline and Configuration
1. Human demonstrations are used to approximate a baseline score—this is a heuristic derived from demo volume but gives a sanity check.
2. The configuration banner shows how many iterations will run, the Monte Carlo budgets (`mc_K`, `mc_H`) and their adaptive steps, the number of self-play workers (`mc_num_workers`), how many MC datasets to reuse, and the evaluation cadence (`fast_eval_episodes` vs `eval_episodes`).
3. Timing instrumentation is enabled via `time.perf_counter()` so each phase reports its wall-clock duration.

## 3. Self-Play Data Generation
1. For each iteration, `generate_self_play_data` produces new Monte Carlo trajectories. It automatically loads `data/policy_mc.pt`, clones environments, and applies `monte_carlo_action`.
2. Parallelism: setting `mc_num_workers > 1` partitions the episode budget across processes. Each worker loads the policy separately, runs `_run_self_play`, and returns its state/action buffer. The parent process concatenates the results and records the elapsed time.
3. MC reuse: every iteration's dataset is archived under `training_logs/mc_demos/iter_<N>.npz`. The latest `mc_reuse_iters` files are concatenated into `data/mc_demos.npz`, preventing catastrophic forgetting and amortizing the cost of expensive rollouts.

## 4. Combined Training
1. Training runs through `train_mc`, which combines human demos (if `include_human_data` remains true) with the aggregated MC data.
2. Adaptive human mixing: once the policy's mean score surpasses the human baseline, imitation demos are dropped automatically to focus on self-play data. This decision is printed so you know when the switch happens.
3. After each training call, the script reports how many seconds the epoch loop consumed.

## 5. Adaptive Evaluation
1. Every iteration begins with a lightweight evaluation (`fast_eval_episodes`) to keep feedback loops tight.
2. The script decides whether to trigger a full evaluation based on three criteria: the iteration number hits `eval_full_every`, the history has no prior score, or the fast eval improved at least `mc_adapt_threshold`. Full evaluations run with `eval_episodes` and log their duration; otherwise, fast stats are reused.
3. Policy regression: if a full or fast eval shows a drop in mean score, the previous checkpoint (backed up before training) is restored, preventing the loop from spiraling downward.
4. Adaptive Monte Carlo budgets: a low-improvement streak counter tracks consecutive iterations whose gains sit below `mc_adapt_threshold`. Once the streak reaches `mc_adapt_patience`, both `mc_K` and `mc_H` are incremented by their step values, growing the search budget.

## 6. Artifact Management
1. Each iteration's policy is saved under `training_logs/models/policy_iter_<N>_mc.pt`, and optional gameplay videos are recorded via `video_recorder.py` if the dependencies are present.
2. Historical metrics (mean score, variance, episode lengths) accumulate in memory and are written to `training_logs/training_history.npz` plus plotted via `visualize_training.py`.
3. The summary table at the end lists every iteration’s score, max score, and incremental improvement so you can audit the entire run after the fact.

## 7. Engineering Notes
1. For long CPU-bound training runs, consider increasing `mc_num_workers`, lowering `mc_episodes`, or adjusting the patience/threshold values to keep the adaptive K/H growth aggressive enough without saturating hardware.
2. To resume a paused run, leave the artifacts in place and rerun `python iterative_train.py`; the bootstrapping logic will select the best checkpoint and continue iterating.
3. When experimenting with curricula (different grid sizes or reward functions), ensure `generate_self_play_data` and `evaluate_policy` share the same parameters so comparisons remain valid.

This script is deliberately verbose—when using text-to-speech, the console logs narrate which stage is executing, how long it takes, and when automatic decisions (like reverting a regression) occur. Feel free to tweak logging verbosity if you plan to run it unattended.
