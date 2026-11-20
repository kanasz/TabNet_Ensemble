#!/bin/bash
# Queues the repeated full-GA measurements as SLURM arrays with a throttle, so they do
# not all run at once: SLURM starts the next run whenever one finishes.
#
# GA-HESO (the expensive one) gets its own array: 1 run per dataset with a 48 h limit,
# all datasets at once. The baselines get 10 runs each with a 24 h limit.
#
#   bash results_analysis/time_complexity/submit_full_ga.sh                      # everything
#   DATASETS="glass_4" bash results_analysis/time_complexity/submit_full_ga.sh   # small dataset only
#   METHODS="ga_heso" bash ...                                                   # GA-HESO only
#   METHODS="selfpe smote_xgboost smote_tabnet" bash ...                         # baselines only
#   RUNS=20 bash ...                     # more baseline repetitions (done runs are kept)
#   GA_HESO_RUNS=3 bash ...              # more GA-HESO repetitions later (run 1 is kept)
#   SEED_MODE=paper bash ...             # every run = the paper's seed 42
#
# Variables:
#   METHODS, DATASETS, SEED_MODE (run|paper)
#   baselines: RUNS (10), MAX_PARALLEL (4), TIME (24:00:00)
#   GA-HESO:   GA_HESO_RUNS (1), GA_HESO_MAX_PARALLEL (3), GA_HESO_TIME (48:00:00)
# A run that does not finish within its limit requeues itself and resumes (see
# run_full_ga_on_hpc.sh). Re-running the same command later is safe: finished runs
# are skipped in seconds.
set -e
METHODS="${METHODS:-selfpe smote_xgboost smote_tabnet ga_heso}"
DATASETS="${DATASETS:-glass_4 yeast_4 abalone19}"
SEED_MODE="${SEED_MODE:-run}"
RUNS="${RUNS:-10}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TIME="${TIME:-24:00:00}"
GA_HESO_RUNS="${GA_HESO_RUNS:-1}"
GA_HESO_MAX_PARALLEL="${GA_HESO_MAX_PARALLEL:-3}"
GA_HESO_TIME="${GA_HESO_TIME:-48:00:00}"

cd "$(dirname "$0")/../.."

submit() {  # methods runs max_parallel time label
    local methods="$1" runs="$2" par="$3" time="$4" label="$5"
    [[ -z "${methods// /}" ]] && return 0
    local n_pairs=$(( $(wc -w <<< "$methods") * $(wc -w <<< "$DATASETS") ))
    local n_tasks=$(( n_pairs * runs ))
    echo "[$label] methods: $methods | datasets: $DATASETS"
    echo "[$label] $n_pairs pairs x $runs run(s) = $n_tasks full GA runs, at most $par at once, time limit $time, seed mode $SEED_MODE"
    sbatch --array=0-$(( n_tasks - 1 ))%"$par" --time="$time" --job-name="full-ga-$label" \
           --export=ALL,METHODS="$methods",DATASETS="$DATASETS",RUNS="$runs",SEED_MODE="$SEED_MODE" \
           results_analysis/time_complexity/run_full_ga_on_hpc.sh
}

baselines=""
heso=""
for m in $METHODS; do
    if [[ "$m" == "ga_heso" ]]; then heso="ga_heso"; else baselines="$baselines $m"; fi
done

submit "$baselines" "$RUNS" "$MAX_PARALLEL" "$TIME" baselines
submit "$heso" "$GA_HESO_RUNS" "$GA_HESO_MAX_PARALLEL" "$GA_HESO_TIME" ga-heso
