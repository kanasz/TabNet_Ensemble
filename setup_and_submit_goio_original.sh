#!/bin/bash
# One-step setup + submission for the GOIO "original way" baseline on the HPC.
# Run it on the LOGIN node from anywhere:
#
#   bash setup_and_submit_goio_original.sh                 # check, verify job, then real run
#   bash setup_and_submit_goio_original.sh --no-verify     # skip the verification job
#   bash setup_and_submit_goio_original.sh --verify-only   # only the verification job
#
# What it does:
#   1. fixes Windows line endings in the .sh files (copied from Windows -> CRLF)
#   2. checks that all files are present and the conda env imports work
#   3. submits a short verification job: test_goio_original_all_datasets.py runs
#      upstream GOIO (split, MLVAE, CLDM train/sample/eval) on fold 0 of all 24
#      datasets with 2 epochs (catches any data-dependent crash)
#   4. submits run_goio_original_on_hpc.sh (6 families in parallel) with
#      --dependency=afterok on the verification job, so the real run starts
#      only if every dataset passed, and is cancelled automatically otherwise
# Nothing heavy runs on the login node itself.

set -euo pipefail

PROJECT_DIR="/mnt/home/pegnke460/TabNet_Ensemble"
CONDA_ENV="pGnip_TabNet"
ACCOUNT="perun26011493"
PARTITION="gpu_long"

MODE="all"
case "${1:-}" in
    "") ;;
    --no-verify) MODE="no-verify" ;;
    --verify-only) MODE="verify-only" ;;
    *) echo "usage: bash $0 [--no-verify | --verify-only]" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"
mkdir -p logs

echo "== 1/4 line endings"
for f in run_goio_original_on_hpc.sh setup_and_submit_goio_original.sh; do
    if grep -q $'\r' "$f"; then
        sed -i 's/\r$//' "$f"
        echo "   fixed CRLF in $f"
    fi
done

echo "== 2/4 files"
REQUIRED=(
    ga_heso_sota_methods/GOIO_ORIGINAL/main.py
    ga_heso_sota_methods/GOIO_ORIGINAL/utils.py
    ga_heso_sota_methods/GOIO_ORIGINAL/utils_train.py
    ga_heso_sota_methods/GOIO_ORIGINAL/data/dataprocessing.py
    ga_heso_sota_methods/GOIO_ORIGINAL/MLVAE/main.py
    ga_heso_sota_methods/GOIO_ORIGINAL/CLDM/evaluation.py
    ga_heso_sota_methods/GOIO_ORIGINAL/src/__init__.py
    optimization/no_ga_goio_original_runner.py
    test_goio_original_all_datasets.py
    run_goio_original_on_hpc.sh
    results_analysis/parse_sota_results.py
    predictions/wine/prediction_wine_goio_original.py
    predictions/yeast/prediction_yeast_goio_original.py
    predictions/glass/prediction_glass_goio_original.py
    predictions/ecoli/prediction_ecoli_goio_original.py
    predictions/abalone/prediction_abalone_goio_original.py
    predictions/synthetic/prediction_synthetic_goio_original.py
)
MISSING=0
for f in "${REQUIRED[@]}"; do
    if [[ ! -f "$f" ]]; then echo "   MISSING: $f" >&2; MISSING=1; fi
done
(( MISSING == 0 )) || { echo "Copy the missing files to $PROJECT_DIR first." >&2; exit 1; }
echo "   all ${#REQUIRED[@]} files present"

echo "== 3/4 conda env and imports"
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found" >&2; exit 1
fi
set +u; source "$CONDA_SH"; conda activate "$CONDA_ENV"; set -u
export PYTHONPATH="$PROJECT_DIR"
python -c "import torch, einops, tqdm, scipy, category_encoders, icecream, tomli, tomli_w, imblearn, sklearn, pandas; print('   torch', torch.__version__, '| sklearn', sklearn.__version__, '| pandas', pandas.__version__, '| category_encoders/icecream/tomli OK')"     || { echo "   GOIO needs: torch, einops, tqdm, scipy, category_encoders, icecream, tomli, tomli-w, imbalanced-learn, scikit-learn, pandas (do NOT install libzero: it pins torch<2)" >&2; exit 1; }
python -c "import base_functions; from optimization.no_ga_goio_original_runner import run_goio_original; print('   project + GOIO runner import OK')"

echo "== 4/4 submit"
VERIFY_ID=""
if [[ "$MODE" != "no-verify" ]]; then
    VERIFY_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=goio-original-verify
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=$ACCOUNT
#SBATCH --output=$PROJECT_DIR/logs/slurm_goio_original_verify_%j.log
#SBATCH --error=$PROJECT_DIR/logs/slurm_goio_original_verify_%j.log
set -e
source "$CONDA_SH"
conda activate $CONDA_ENV
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR PYTHONUNBUFFERED=1
python -u test_goio_original_all_datasets.py
EOF
)
    echo "   verification job: $VERIFY_ID   (log: logs/slurm_goio_original_verify_${VERIFY_ID}.log)"
fi

if [[ "$MODE" != "verify-only" ]]; then
    DEP=()
    if [[ -n "$VERIFY_ID" ]]; then
        DEP=(--dependency="afterok:$VERIFY_ID" --kill-on-invalid-dep=yes)
    fi
    RUN_ID=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} run_goio_original_on_hpc.sh)
    echo "   real run (array 0-5): $RUN_ID   (logs: logs/slurm_goio_original_${RUN_ID}_<0-5>.log)"
    [[ -n "$VERIFY_ID" ]] && echo "   it starts only after verification $VERIFY_ID succeeds"
fi

cat <<EOF

Monitor:
  squeue -u \$USER
  tail -n 30 logs/slurm_goio_original_verify_${VERIFY_ID:-<id>}.log        # PASS/FAIL table at the end
  grep -hE "fold [0-9]/5|evaluated fitness|Error|Traceback" logs/slurm_goio_original_${RUN_ID:-<id>}_*.log

When all six families have finished:
  python results_analysis/parse_sota_results.py
EOF
