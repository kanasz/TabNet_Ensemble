#SBATCH --account=p482-24-1

source venv_py_310/bin/activate

python boosting_tabnet/prediction_abalone.py