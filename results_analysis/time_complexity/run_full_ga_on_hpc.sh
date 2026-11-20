#!/bin/bash

#SBATCH --job-name=full-ga-time
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=perun26011493
# One array task = ONE full GA run of one (method, dataset). Submit with
# submit_full_ga.sh, which sets --array and the throttle (%N = at most N at once).
#SBATCH --output=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_full_ga_%A_%a.log
#SBATCH --error=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_full_ga_%A_%a.log
#SBATCH --open-mode=append
# Auto-continue: 10 min before the time limit SLURM sends USR1 to this script, which
# puts the task back in the queue; when it starts again, measure_full_ga.py resumes
# the GA from its last generation (timing stays correct, see measure_full_ga.py).
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

# MEASURED cost of the whole GA search, repeated: re-runs the paper's GA from scratch,
# timed (results_analysis/time_complexity/measure_full_ga.py). Submit through
#   bash results_analysis/time_complexity/submit_full_ga.sh
# Task index i -> run = i / n_pairs + 1, pair = i % n_pairs: run 1 of every pair
# first, then run 2 ..., so all pairs have results early and machine drift is spread.
# Runs longer than the time limit continue automatically (requeue, see above). If a
# task was killed some other way (node failure, scancel), resubmitting the same
# submit command resumes it; finished runs are skipped.

set -e

PROJECT_DIR="/mnt/home/pegnke460/TabNet_Ensemble"
CONDA_ENV="pGnip_TabNet"
RUNS="${RUNS:-10}"
SEED_MODE="${SEED_MODE:-run}"          # run = different search per run; paper = seed 42 always
METHODS="${METHODS:-selfpe smote_xgboost smote_tabnet ga_heso}"
DATASETS="${DATASETS:-glass_4 yeast_4 abalone19}"

PAIRS=()
for d in $DATASETS; do for m in $METHODS; do PAIRS+=("$m $d"); done; done
N_PAIRS=${#PAIRS[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= N_PAIRS * RUNS )); then
    echo "Error: array index $TASK_ID out of range 0-$(( N_PAIRS * RUNS - 1 ))" >&2
    exit 1
fi
RUN=$(( TASK_ID / N_PAIRS + 1 ))
read -r METHOD DATASET <<< "${PAIRS[$(( TASK_ID % N_PAIRS ))]}"
SEED_ARG=()
[[ "$SEED_MODE" == "run" ]] && SEED_ARG=(--seed run)

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: could not find the Conda initialization file conda.sh" >&2
    exit 1
fi
conda activate "$CONDA_ENV"

echo "========================================"
echo "Job ${SLURM_JOB_ID:-local} task $TASK_ID: full GA of $METHOD on $DATASET, run $RUN/$RUNS, seed mode $SEED_MODE"
echo "Host: $(hostname)   CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: no GPU visible"
echo "Started: $(date)   (start no. ${SLURM_RESTART_COUNT:-0} of this task)"
echo "========================================"

requeue() {
    echo "$(date): time limit near - requeueing, the GA resumes from its last generation"
    scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
        || scontrol requeue "$SLURM_JOB_ID"
}
trap requeue USR1

# in the background + wait, so the USR1 trap runs as soon as the signal arrives
python -u results_analysis/time_complexity/measure_full_ga.py --method "$METHOD" --dataset "$DATASET" \
    --run "$RUN" "${SEED_ARG[@]}" &
PID=$!
set +e
wait "$PID"
STATUS=$?
if (( STATUS > 128 )); then   # wait was interrupted by the signal: the GA is still running
    wait "$PID"               # until SLURM ends the task at the limit (then it is requeued)
    STATUS=$?
fi
echo "Finished: $(date)  exit $STATUS"
exit "$STATUS"
