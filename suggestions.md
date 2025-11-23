# Suggestions for CPU-Only Iterative Training

## Observations
- Adaptive K/H budgets, MC demo reuse, fast/full evaluation schedules, timing logs, and parallel self-play are now implemented, so the main remaining inefficiencies live inside the Monte Carlo rollouts themselves and in how training data is mixed.
- Human demonstrations are currently removed from the mix as soon as the baseline is beaten; this avoids regressions but may allow the agent to forget basic safety strategies over very long runs.
- The aggregation of MC demos loads all files into memory each iteration, which may become a limiting factor when retaining many iterations.

## Improvement Ideas (Highest Impact First)
1. **Rollout Early Termination**: extend `monte_carlo_action` to prune rollouts that can no longer surpass the best action’s score (e.g., use remaining horizon * max reward heuristic). This directly cuts CPU time spent simulating hopeless branches.
2. **Occasional Human Replay Injection**: after switching to MC-only updates, periodically re-add a small fraction of human demos (e.g., 5% every 3 iterations) with a high weight. This keeps the agent grounded in safe behaviors without meaningfully slowing training.
3. **Curriculum Grid Sizes**: start self-play on smaller grids (6×6, 8×8) and step up to 10×10 once the mean score crosses thresholds. Shorter episodes dramatically reduce MC cost in the early phases while still teaching core mechanics.
4. **Dynamic Learning Rate / Epoch Scheduler**: monitor `train_mc` validation accuracy and automatically reduce epochs or learning rate once improvements plateau. Saves CPU time late in iterative runs when large data mixes make extra epochs wasteful.
5. **Incremental MC Dataset Loading**: when aggregating multiple `training_logs/mc_demos/iter_*.npz` files, use memory-mapped arrays or chunked reading so the process doesn’t require loading every file into RAM. This keeps memory use predictable during very long runs.
6. **Per-action Monte Carlo Budgeting**: instead of increasing `K`/`H` uniformly, assign more rollouts to actions with closer value estimates (high variance) and fewer to clearly inferior ones. This focuses CPU time where Monte Carlo estimates are most uncertain.
7. **Parallel Evaluation**: leverage the new multiprocessing helpers to evaluate policies across multiple processes, allowing the “fast eval” and “full eval” passes to scale with available CPU cores.
