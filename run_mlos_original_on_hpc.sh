#!/bin/bash

#SBATCH --job-name=mlos-original
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --account=perun26011493
# One array task per dataset family (index into SCRIPTS below).
#SBATCH --array=0-5
# SLURM opens these files before the job starts, so the logs directory must
# already exist at submit time. Absolute paths, because a relative one
# resolves against the submit-time working directory, not PROJECT_DIR.
#SBATCH --output=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_mlos_original_%A_%a.log
#SBATCH --error=/mnt/home/pegnke460/TabNet_Ensemble/logs/slurm_mlos_original_%A_%a.log

# MLOS run the original way (no GA): upstream MLOS.fit_Sample from the
# untouched copy in ga_heso_sota_methods/MLOS_ORIGINAL on each training fold,
# then 5 classifiers x 10 repeats, see optimization/no_ga_mlos_original_runner.py.
# MLOS is CPU-only (numpy / scikit-learn); the GPU partition is used only to
# match the other baselines' submission setup.
#
#   sbatch run_mlos_original_on_hpc.sh             # all six families
#   sbatch --array=2 run_mlos_original_on_hpc.sh   # only glass
#
# Index: 0 wine, 1 yeast, 2 glass, 3 ecoli, 4 abalone, 5 synthetic.
# Per fold: 10 MLOS draws on the training part, each evaluated with
# GaussianNB, DecisionTree, SVC, RandomForest and KNN on the test part.

set -e

PROJECT_DIR="/mnt/home/pegnke460/TabNet_Ensemble"
CONDA_ENV="pGnip_TabNet"

SCRIPTS=(
    "predictions/wine/prediction_wine_mlos_original.py"
    "predictions/yeast/prediction_yeast_mlos_original.py"
    "predictions/glass/prediction_glass_mlos_original.py"
    "predictions/ecoli/prediction_ecoli_mlos_original.py"
    "predictions/abalone/prediction_abalone_mlos_original.py"
    "predictions/synthetic/prediction_synthetic_mlos_original.py"
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

# Fail fast if the upstream copy or its dependencies are missing. MLOS.py
# imports numpy, scipy, scikit-learn and matplotlib (pyplot, never shown).
if [[ ! -f "$PROJECT_DIR/ga_heso_sota_methods/MLOS_ORIGINAL/MLOS.py" ]]; then
    echo "Error: ga_heso_sota_methods/MLOS_ORIGINAL/MLOS.py not found" >&2
    exit 1
fi
if ! python -c "import numpy, scipy, sklearn, pandas, matplotlib, imblearn; print('numpy', numpy.__version__, '| sklearn', sklearn.__version__, '| pandas', pandas.__version__)"; then
    echo "Error: MLOS dependencies missing (numpy, scipy, scikit-learn, pandas, matplotlib, imbalanced-learn)" >&2
    exit 1
fi
python -c "import base_functions; print('base_functions:', base_functions.__file__)"
python -c "from optimization.no_ga_mlos_original_runner import run_mlos_original, load_mlos; load_mlos(); print('MLOS original runner: OK')"

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
