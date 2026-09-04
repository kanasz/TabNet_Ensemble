#!/bin/bash

#SBATCH --job-name=cco-ga
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=perun26011493
# One array task per domain - see the note on cost below.
#SBATCH --array=0-4
# SLURM opens these files before the job starts, so the logs directory must
# already exist at submit time. %A is the array job id, %a the task index.
#SBATCH --output=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_%A_%a.log
#SBATCH --error=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_%A_%a.log

# GA-tuned CCO. Unlike the fixed-default baseline, this searches
# constants.genes_cco (k, beta, t, focal gamma, batch_size) with
# GASotaRunConfig = 30 generations x 30 population = ~900 fitness evaluations
# per dataset, each of which trains the model on 5 folds (joblib n_jobs=5).
#
# Measured reference: the baseline needs ~235 s for one wine dataset (5 folds
# x 100 epochs, sequential). One GA evaluation is ~5 folds in parallel x 50
# epochs, so roughly 15-25 s, which puts a single dataset in the hours and a
# whole domain well past a 12 h wall clock. Hence the job array: each task
# takes one domain, and the tasks run concurrently instead of one 5-day job.
#
# If a task still runs out of time, just resubmit it. The tuner checkpoints
# every generation to ga_logs/cco/{dataset}/checkpoint.pkl and resumes from
# there, so an interrupted dataset continues where it stopped and already
# finished ones fast-forward through a single final evaluation.
#
# Set FRESH=1 to discard those checkpoints first and start the domain over:
#     FRESH=1 sbatch run_cco_ga_on_hpc.sh
# Do that whenever the previous attempt's settings changed, otherwise the old
# checkpoint (with its own num_generations) is what actually gets replayed.

set -e

PROJECT_DIR="/mnt/home/pegnke460/TabNet_Ensemble"
CONDA_ENV="pGnip_TabNet"

# One entry per array index; each runs every dataset of that domain.
SCRIPTS=(
    "predictions/wine/prediction_wine_cco.py"
    "predictions/yeast/prediction_yeast_cco.py"
    "predictions/glass/prediction_glass_cco.py"
    "predictions/ecoli/prediction_ecoli_cco.py"
    "predictions/abalone/prediction_abalone_cco.py"
)

# Outside SLURM (or without --array) run the whole list sequentially.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SELECTED=("${SCRIPTS[$SLURM_ARRAY_TASK_ID]}")
else
    SELECTED=("${SCRIPTS[@]}")
fi

# === Project directory ===

cd "$PROJECT_DIR"
mkdir -p logs

# Add the project root to Python's module search path.
# This allows imports such as: import base_functions
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Print Python output immediately to the SLURM log.
export PYTHONUNBUFFERED=1

# === Activate Conda environment ===

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: could not find the Conda initialization file conda.sh" >&2
    exit 1
fi

conda activate "$CONDA_ENV"

# === Environment information ===

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}  Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: $(command -v python)"
echo "Python version: $(python --version 2>&1)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Scripts in this task: ${SELECTED[*]}"
echo "FRESH: ${FRESH:-0}"
echo "========================================"

# Verify that the project module and the vendored CCO code are importable.
python -c \
    "import base_functions; print('base_functions:', base_functions.__file__)"
python -c \
    "from ga_heso_sota_methods.CCO.utils import Cluster; print('CCO utils: OK')"
python -c \
    "from optimization.ga_cco_tuner import GaCCOTuner; print('GA CCO tuner: OK')"

# === Run Python scripts sequentially ===

# One dataset family failing should not cost the others in this task, so
# failures are collected and reported at the end (set -e is suspended for the
# loop only).
FAILED=()
set +e

for SCRIPT in "${SELECTED[@]}"; do
    SCRIPT_PATH="$PROJECT_DIR/$SCRIPT"

    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "Error: script not found: $SCRIPT_PATH" >&2
        FAILED+=("$SCRIPT (missing)")
        continue
    fi

    SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

    if [[ "${FRESH:-0}" == "1" ]]; then
        echo "FRESH=1: removing $SCRIPT_DIR/ga_logs/cco"
        rm -rf "$SCRIPT_DIR/ga_logs/cco"
    fi

    echo "========================================"
    echo "Starting: $SCRIPT"
    echo "Time: $(date)"
    echo "========================================"

    # The prediction scripts resolve their output paths against their own
    # folder ('results/...', 'ga_logs/...'), so each one has to run from the
    # directory it lives in. Imports still work because PYTHONPATH points at
    # the project root.
    cd "$SCRIPT_DIR"
    python -u "$(basename "$SCRIPT_PATH")"
    STATUS=$?
    cd "$PROJECT_DIR"

    if [[ $STATUS -ne 0 ]]; then
        echo "FAILED (exit $STATUS): $SCRIPT" >&2
        FAILED+=("$SCRIPT (exit $STATUS)")
    fi

    echo "========================================"
    echo "Finished: $SCRIPT"
    echo "Time: $(date)"
    echo "========================================"
done

set -e

# === Summary ===

if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All scripts completed successfully."
else
    echo "Completed with ${#FAILED[@]} failure(s):" >&2
    for ITEM in "${FAILED[@]}"; do
        echo "  - $ITEM" >&2
    done
    exit 1
fi