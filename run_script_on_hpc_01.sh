#!/bin/bash

#SBATCH --job-name=python-scripts
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=perun26011493
# SLURM opens these files before the job starts, so the logs directory must
# already exist at submit time - the mkdir below is too late for them. The
# paths are absolute because a relative one resolves against the submit-time
# working directory, not PROJECT_DIR.
#SBATCH --output=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_%j.log
#SBATCH --error=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_%j.log

set -e

PROJECT_DIR="/mnt/home/pegnke460/TabNet_Ensemble"
CONDA_ENV="pGnip_TabNet"

# Scripts run sequentially in this order.
SCRIPTS=(
    "predictions/wine/prediction_wine_oc_tabnet_ensemble_no_clustering.py"
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

# Verify that the project module is importable.
python -c \
    "import base_functions; print('base_functions:', base_functions.__file__)"

# === GPU monitoring ===

JOB_ID="${SLURM_JOB_ID:-local}"
GPU_LOG="$PROJECT_DIR/logs/gpu_usage_${JOB_ID}.log"
NVIDIA_PID=""

cleanup() {
    if [[ -n "$NVIDIA_PID" ]] &&
       kill -0 "$NVIDIA_PID" 2>/dev/null; then
        echo "Stopping GPU monitoring..."
        kill "$NVIDIA_PID" 2>/dev/null || true
        wait "$NVIDIA_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Starting GPU monitoring: $GPU_LOG"

    nvidia-smi \
        --query-gpu=timestamp,name,index,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv \
        -l 60 > "$GPU_LOG" 2>&1 &

    NVIDIA_PID=$!
else
    echo "Warning: nvidia-smi was not found; GPU monitoring is disabled." >&2
fi

# === Run Python scripts sequentially ===

for SCRIPT in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="$PROJECT_DIR/$SCRIPT"

    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "Error: script not found: $SCRIPT_PATH" >&2
        exit 1
    fi

    echo "========================================"
    echo "Starting: $SCRIPT"
    echo "Time: $(date)"
    echo "========================================"

    # The prediction scripts resolve paths against their own folder - e.g.
    # get_config_files("../../models/configurations") and the 'results/...'
    # output dirs - so each one has to run from the directory it lives in.
    # Imports still work because PYTHONPATH points at the project root.
    cd "$(dirname "$SCRIPT_PATH")"
    python -u "$(basename "$SCRIPT_PATH")"
    cd "$PROJECT_DIR"

    echo "========================================"
    echo "Finished: $SCRIPT"
    echo "Time: $(date)"
    echo "========================================"
done

echo "All scripts completed successfully."
