#!/bin/bash

#SBATCH --job-name=dgot-baseline
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=perun26011493
# One array task per dataset family (index into SCRIPTS below), so the six
# families run in parallel and each gets its own 48h limit.
#SBATCH --array=0-5
# SLURM opens these files before the job starts, so the logs directory must
# already exist at submit time. Absolute paths, because a relative one
# resolves against the submit-time working directory, not PROJECT_DIR.
#SBATCH --output=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_dgot_%A_%a.log
#SBATCH --error=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_dgot_%A_%a.log

# Fixed-default (no GA) DGOT baseline: the published pipeline from the
# untouched upstream copy in ga_heso_sota_methods/DGOT_BASELINE, see
# optimization/no_ga_dgot_runner.py.
#
#   sbatch run_dgot_baseline_on_hpc.sh             # all six families
#   sbatch --array=2 run_dgot_baseline_on_hpc.sh   # only glass
#
# A rerun skips folds that already finished with the same data and settings
# (cache in ga_heso_sota_methods/DGOT_BASELINE/saved_log/runner/), so a job
# that hit the time limit can simply be resubmitted.

set -e

PROJECT_DIR="/mnt/home/pegnke460/TabNet_Ensemble"
CONDA_ENV="pGnip_TabNet"

SCRIPTS=(
    "predictions/wine/prediction_wine_dgot_baseline.py"
    "predictions/yeast/prediction_yeast_dgot_baseline.py"
    "predictions/glass/prediction_glass_dgot_baseline.py"
    "predictions/ecoli/prediction_ecoli_dgot_baseline.py"
    "predictions/abalone/prediction_abalone_dgot_baseline.py"
    "predictions/synthetic/prediction_synthetic_dgot_baseline.py"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -n "$TASK_ID" ]]; then
    if (( TASK_ID < 0 || TASK_ID >= ${#SCRIPTS[@]} )); then
        echo "Error: array index $TASK_ID out of range 0-$(( ${#SCRIPTS[@]} - 1 ))" >&2
        exit 1
    fi
    SCRIPTS=("${SCRIPTS[$TASK_ID]}")
fi
# Without an array index (e.g. run via `bash` on an interactive node) every
# family runs sequentially.

# === Project directory ===

cd "$PROJECT_DIR"
mkdir -p logs

# Project root on the module search path, for `import base_functions` etc.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Print Python output immediately to the SLURM log.
export PYTHONUNBUFFERED=1
# DGOT's train.py redraws a tqdm bar every iteration; refresh it at most
# every 30 s so the log stays readable (tqdm >= 4.66).
export TQDM_MININTERVAL=30

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
echo "Job ID: ${SLURM_JOB_ID:-local}  array task: ${TASK_ID:-none}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: $(command -v python)"
echo "Python version: $(python --version 2>&1)"
echo "PYTHONPATH: $PYTHONPATH"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: no GPU visible"
echo "========================================"

# Fail fast if the upstream copy or its dependencies are missing. DGOT's
# train.py and evaluation.py import xgboost, yaml and einops at module level
# (einops and yaml are missing from upstream's requirements.txt). Everything
# needed is listed in dgot_requirements.txt; install it once into the env:
#   conda activate pGnip_TabNet && pip install -r dgot_requirements.txt
if [[ ! -f "$PROJECT_DIR/ga_heso_sota_methods/DGOT_BASELINE/train.py" ]]; then
    echo "Error: ga_heso_sota_methods/DGOT_BASELINE/train.py not found" >&2
    exit 1
fi
if ! python -c "import torch, xgboost, yaml, einops, pandas, imblearn; assert int(pandas.__version__.split('.')[0]) < 3, 'pandas>=3 breaks DGOT'; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| xgboost', xgboost.__version__, '| pandas', pandas.__version__)"; then
    echo "Error: DGOT dependencies missing or incompatible - run: pip install -r $PROJECT_DIR/dgot_requirements.txt" >&2
    exit 1
fi
python -c "import base_functions; print('base_functions:', base_functions.__file__)"
python -c "from optimization.no_ga_dgot_runner import run_dgot_baseline, _upstream; _upstream(); print('DGOT runner + upstream modules: OK')"

# === Run Python scripts sequentially ===

# One family failing should not cost the others (when run without an array),
# so failures are collected and reported at the end.
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
