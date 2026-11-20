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
python results_analysis/time_complexity/check_timing_runs.py --logs "logs/slurm_time_*.log"
python results_analysis/time_complexity/ga_convergence.py     # -> summary/ (reads GA .pkl, needs pygad) - first
python results_analysis/time_complexity/summarize_timing.py   # -> summary/
```

## Measured full GA search (`measure_full_ga.py`, `run_full_ga_on_hpc.sh`)
Re-runs the paper's GA from scratch (settings taken from the paper checkpoint, the tuner's own
`run_experiment`), once per method x dataset, into `work/full_ga/` - never resuming, never
touching the paper's checkpoints. Times the whole search, every fitness evaluation and each
generation -> `raw_full_ga/`. `summarize_timing.py` then uses the measured search time instead
of the estimate and reports the estimate's error (`cost_split.csv`).
```bash
python results_analysis/time_complexity/measure_full_ga.py --method ga_heso --dataset abalone19 --dry-run
# baselines: 10 runs per method x dataset, 24 h limit, at most 4 at once
# GA-HESO:   1 run per dataset, 48 h limit, all 3 at once (GA_HESO_RUNS / GA_HESO_TIME)
# each run uses its own GA seed (1000 + run); SEED_MODE=paper keeps the paper's 42
bash results_analysis/time_complexity/submit_full_ga.sh
DATASETS="glass_4" bash results_analysis/time_complexity/submit_full_ga.sh    # one dataset only
```
Re-running a submit command later is safe: finished runs are skipped.

**Time limits (e.g. 24 h):** runs longer than the limit continue on their own. 10 min before
the limit the task requeues itself (`--requeue --signal=B:USR1@600`) and, when it starts
again, resumes the GA from its last generation with the saved RNG state - the same search as
an uninterrupted run. Timing counts each attempt only up to its last checkpoint; work lost at
the kill (`lost_seconds`) and re-evaluations after loading (`resume_overhead_seconds`) are
reported separately, and `search_seconds_uninterrupted` is the number to report.

## Analyses for the paper
- **Cost split** (`cost_split.csv`, `table_cost_split.tex`): (a) one-off search cost
  (hours, CPU-core-hours, GPU-hours) vs. (b) final model (training s, prediction ms/1000, RAM).
- **Accuracy vs. time** (`fig_accuracy_vs_time.png`): G-mean vs. time, filled marker = final
  training, hollow = GA search + final training. This experiment does not measure accuracy:
  put the paper's G-mean values into `paper_gmean.csv` (`method,dataset,gmean`); otherwise the
  GA-reported fitness / the timed SOTA run is used (`accuracy_used_in_figure.csv` says which).
- **GA convergence** (`convergence.csv`, `fig_convergence.png`): generation at which the GA
  reaches 90/95/99 % of its final G-mean (`value`) and of its improvement over the initial
  population (`improvement`, the honest one), with the share of search time that could be saved.
Interrupted runs resume where they stopped; failed runs leave `raw/.../run_NN.json.error`.
GA-HESO requires CUDA (its losses are CUDA-only).
- **GA-tuned vs. SOTA** (`ga_vs_sota.csv`, `table_ga_vs_sota.tex`, `fig_time_to_quality.png`),
  for every GA method x SOTA method x dataset:
  - *A, untuned*: the paper GA's random first generation = the method without tuning. Share of
    those configurations already reaching the SOTA G-mean; cost = one fitness evaluation.
  - *C, time-to-quality*: GA search time until the best-so-far G-mean reaches the SOTA G-mean
    (hours and x SOTA training time), and the best G-mean reached within the SOTA training time.
    Uses the measured full GA runs; without them, the paper checkpoint's curve with the
    estimated time (`curve_source` says which).
