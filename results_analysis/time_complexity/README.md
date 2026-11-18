# Time complexity of GA-HESO vs. baselines and SOTA

20 independent timed runs (+1 warm-up, excluded) per method × dataset; each run is a
separate Python process.

| Kind | Methods | What one run times |
|---|---|---|
| GA-tuned (paper) | GA-HESO, SMOTE+TabNet, SelfPE, SMOTE+XGBoost | Retraining of the **best chromosome** from the paper's GA checkpoint (`.pkl`) on the same 5 folds, same seeds and code as the GA fitness function. Phases are timed separately; the full GA search cost is **estimated** = one fitness evaluation × number of evaluations. |
| SOTA (untuned) | DGOT, SOS, MLOS (CCO, GOIO available) | The upstream pipeline via `optimization/no_ga_*_runner.py`, `resume=False`. Data go under the name `timing_<dataset>`, so the paper's caches are never touched. |

Datasets: `glass_4` (214 rows), `yeast_4` (1484), `abalone19` = A4 (4174).
Everything is configured in `timing_config.py`.

## Measured per run
- wall-clock time (5 folds), training time, prediction time (+ per 1000 samples), time per fold
- phases: setup (GA-HESO MeanShift bandwidths), preprocess, resample, fit, predict; for SOTA, the runner's own per-fold phases (train / eval / fine_tune / oversample ...)
- CPU time of the whole process tree, CPU/wall ratio (effective parallelism)
- peak RAM (process tree), peak GPU memory and mean GPU utilisation (needs `pynvml`)
- model size: ensemble members, #parameters, epochs actually trained, boosting rounds
- GA search estimate (nominal and distinct evaluations; for GA-HESO also with 5 folds in parallel, as in the GA)
- G-mean as a sanity check against the value the GA reported
- environment: host, CPU, GPU, library versions, thread settings

## Statistics over the 20 runs (`summarize_timing.py`)
n, mean, std, median, min, max, Q1, Q3, IQR, CV %, 95 % CI of the mean, Tukey outliers,
ratio to GA-HESO, and the empirical scaling exponent b in time ~ n^b.

## Usage
```bash
# HPC: one array task per method, all on one node type (see comments in the script)
sbatch results_analysis/time_complexity/run_time_complexity_on_hpc.sh
# or directly
python results_analysis/time_complexity/run_timing_experiment.py --dry-run
python results_analysis/time_complexity/run_timing_experiment.py --methods ga_heso --runs 20
# afterwards
python results_analysis/time_complexity/summarize_timing.py   # -> summary/
```
Interrupted runs resume where they stopped; failed runs leave `raw/.../run_NN.json.error`.
GA-HESO requires CUDA (its losses are CUDA-only).
