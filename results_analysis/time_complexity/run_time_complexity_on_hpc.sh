#!/bin/bash

#SBATCH --job-name=time-complexity
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=perun26011493
# No --exclusive: on this cluster that requires all 8 GPUs of the node. The task
# still owns its GPU and its CPU cores (cgroups); only node-wide resources (memory
# bandwidth, disk) are shared with other jobs. The 20 interleaved runs and the
# median/IQR in the summary absorb that noise.
# One array task per method (index into METHODS below).
#SBATCH --array=0-6
#SBATCH --output=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_time_%A_%a.log
#SBATCH --error=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_time_%A_%a.log

# Time-complexity measurement: 20 independent timed runs (+1 warm-up) of each method
# on each dataset, see results_analysis/time_complexity/README.md.
#
#   sbatch results_analysis/time_complexity/run_time_complexity_on_hpc.sh            # all methods
#   sbatch --array=0 results_analysis/time_complexity/run_time_complexity_on_hpc.sh  # only GA-HESO
#   RUNS=2 WARMUP=0 sbatch --array=0-6 --time=04:00:00 ...                           # quick trial
#
# Every array task gets its own node, so to keep the times comparable all tasks
# must land on the same node TYPE (same CPU and GPU model). Pin a node list or a
# feature if the partition is heterogeneous, e.g.
#   sbatch --constraint=<gpu-feature> ...   or   sbatch --nodelist=<node> ...
# summarize_timing.py warns when the runs come from different hardware.
#
# Resumable: finished runs are skipped, so a task that hit the time limit can
# simply be resubmitted with the same command.

set -e

PROJECT_DIR="/mnt/home/pegnke460/TabNet_Ensemble"
CONDA_ENV="pGnip_TabNet"
RUNS="${RUNS:-20}"
WARMUP="${WARMUP:-1}"
DATASETS="${DATASETS:-glass_4 yeast_4 abalone19}"

METHODS=(
    ga_heso
    smote_tabnet
    selfpe
    smote_xgboost
    dgot_baseline
    sos_original
    mlos_original
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -n "$TASK_ID" ]]; then
    if (( TASK_ID < 0 || TASK_ID >= ${#METHODS[@]} )); then
        echo "Error: array index $TASK_ID out of range 0-$(( ${#METHODS[@]} - 1 ))" >&2
        exit 1
    fi
    METHODS=("${METHODS[$TASK_ID]}")
fi

cd "$PROJECT_DIR"
mkdir -p logs

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=30
# Same thread budget for every method = the CPUs SLURM gave the task.
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
echo "Job ID: ${SLURM_JOB_ID:-local}  array task: ${TASK_ID:-none}  methods: ${METHODS[*]}"
echo "Host: $(hostname)"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: no GPU visible"
echo "Python: $(command -v python) $(python --version 2>&1)"
echo "Runs: $RUNS (+ $WARMUP warm-up)  datasets: $DATASETS"
echo "========================================"

# psutil is required (RAM/CPU sampling); pynvml (pip install nvidia-ml-py) is optional
# and only adds GPU memory/utilisation.
python -c "import psutil; print('psutil', psutil.__version__)" || {
    echo "Error: psutil missing - pip install psutil" >&2; exit 1; }
python -c "import pynvml; pynvml.nvmlInit(); print('pynvml OK')" || \
    echo "WARNING: pynvml unavailable - GPU memory/utilisation will not be recorded (pip install nvidia-ml-py)"

python -u results_analysis/time_complexity/run_timing_experiment.py \
    --methods "${METHODS[@]}" \
    --datasets $DATASETS \
    --runs "$RUNS" \
    --warmup "$WARMUP"
