# Suggestions for CPU-Only Iterative Training

## Observations
- `iterative_train.py` re-evaluates the policy for every iteration with fixed `eval_episodes` (default 50), fixed MC rollout budgets (`mc_K=5`, `mc_H=20`), and regenerates the full `mc_demos.npz` each loop. On a CPU, this becomes the dominant cost because each `monte_carlo_action` in `self_play.py` performs `K * H` environment steps per candidate action.
- Human demonstrations remain on disk but, once the human baseline is beaten, they are excluded entirely. This means the model stops seeing any low-level corrective supervision even though those samples are cheap to reuse.
- Evaluation/instrumentation is spread across several scripts, but no profiling exists to highlight which stage is the current bottleneck.

## Improvement Ideas
1. **Batching Monte Carlo Evaluations**: `monte_carlo_action` currently evaluates each rollout sequentially, repeatedly calling the policy on small batches. Group the states from all candidate actions/rollouts into one tensor so PyTorch can leverage vectorization and reduce per-call overhead.
2. **Cached Policy Outputs**: within a single environment step MC rollouts often revisit identical states. Cache policy logits for each unique state to avoid redundant forward passes, further lowering CPU cost per Monte Carlo decision.
3. **Parallel Episodic Generation**: use `multiprocessing` to run several `SnakeEnv` instances concurrently inside `generate_self_play_data`. Even on CPU-only machines this hides Python overhead and shortens wall-clock time for producing the MC dataset.
4. **Rollout Early Termination**: detect when a rollout cannot beat the current best action (e.g., based on max attainable reward) and terminate it early. Pruning hopeless branches prevents wasting hundreds of steps per action evaluation.
5. **Lightweight Evaluation Schedule**: run a quick evaluation (≈10 episodes) every iteration and only trigger the full 50-episode evaluation when the lightweight run shows ≥0.2 improvement. This keeps regression checks intact while cutting CPU time spent in `evaluate_policy`.
6. **Occasional Human Replay Mixing**: after switching to MC-only training, periodically reintroduce a small slice of human demos (say 5% of each batch) with a strong weight. This guards against the agent forgetting basic safe behaviors without collecting new data.
7. **Curriculum Grid Sizes**: start training on smaller grids (6×6 or 8×8) to speed up rollouts and gradually scale to 10×10 once the agent exceeds certain scores. Shorter episodes reduce MC time yet still teach core mechanics.
8. **Dynamic Learning Rates/Epochs**: monitor validation accuracy in `train_mc` and automatically cut epochs or learning rate when improvements fall below a threshold. This prevents wasting CPU cycles on extra epochs that produce no generalization gains.
9. **Profiled Logging**: wrap data generation, training, and evaluation sections in `time.perf_counter()` timers and log the durations. Identifying the true bottleneck helps decide whether to lower MC parameters, reduce epochs, or parallelize generation.
10. **Incremental Dataset Loading**: when aggregating many historical MC demos, switch to memory-mapped arrays or chunked loading so the pipeline doesn’t require all samples in RAM at once. This keeps the process stable on machines with limited memory while still benefiting from larger datasets.
