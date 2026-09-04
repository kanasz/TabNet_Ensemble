#!/bin/bash

#SBATCH --job-name=cco-baseline
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --account=perun26011493
# SLURM opens these files before the job starts, so the logs directory must
# already exist at submit time. The paths are absolute because a relative one
# resolves against the submit-time working directory, not PROJECT_DIR.
#SBATCH --output=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_%j.log
#SBATCH --error=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_%j.log

# Fixed-default (no GA) CCO baseline over every dataset. CCO is cheap - its
# clustering is O(n^2) but pinned to CPU, and the model is a 3-layer MLP - so
# this needs far less time and memory than the GA-HESO job. The GPU request is
# only there because gpu_short is the partition known to work with this
# account; a CPU partition would do, since Cluster() and synthetic_generation()
# run on CPU regardless.

set -e

PROJECT_DIR="/mnt/home/pegnke460/TabNet_Ensemble"
CONDA_ENV="pGnip_TabNet"

# Scripts run sequentially in this order.
SCRIPTS=(
    "predictions/wine/prediction_wine_cco_baseline.py"
    "predictions/yeast/prediction_yeast_cco_baseline.py"
    "predictions/glass/prediction_glass_cco_baseline.py"
    "predictions/ecoli/prediction_ecoli_cco_baseline.py"
    "predictions/abalone/prediction_abalone_cco_baseline.py"
)

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
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: $(command -v python)"
echo "Python version: $(python --version 2>&1)"
echo "PYTHONPATH: $PYTHONPATH"
echo "========================================"

# Verify that the project module and the vendored CCO code are importable.
python -c \
    "import base_functions; print('base_functions:', base_functions.__file__)"
python -c \
    "from ga_heso_sota_methods.CCO.utils import Cluster; print('CCO utils: OK')"
python -c \
    "from optimization.no_ga_cco_runner import run_cco_baseline; print('CCO runner: OK')"

# === Run Python scripts sequentially ===

# One dataset family failing should not cost the other four, so failures are
# collected and reported at the end instead of aborting the job (set -e is
# suspended for the loop only).
FAILED=()
set +e

for SCRIPT in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="$PROJECT_DIR/$SCRIPT"

    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "Error: script not found: $SCRIPT_PATH" >&2
        FAILED+=("$SCRIPT (missing)")
        continue
    fi

    echo "========================================"
    echo "Starting: $SCRIPT"
    echo "Time: $(date)"
    echo "========================================"

    # The prediction scripts resolve their output paths against their own
    # folder ('results/...'), so each one has to run from the directory it
    # lives in. Imports still work because PYTHONPATH points at the project
    # root.
    cd "$(dirname "$SCRIPT_PATH")"
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