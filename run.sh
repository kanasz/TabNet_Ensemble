#!/bin/sh
#SBATCH --account=p482-24-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -p gpu
#SBATCH -G 1
module load cuda/12.0.1
module load Python/3.10.8-GCCcore-12.2.0
source /projects/p482-24-1/venv/venv_py_310/bin/activate
export PYTHONUNBUFFERED=TRUE
python prediction_ecoli_0_vs_1.py
deactivate