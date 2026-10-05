#!/bin/bash
# Example SLURM job (adapt account, modules and environment to your cluster).
#SBATCH --job-name=ssr_alc_exp4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
##SBATCH --account=<your-account>

# module load cuda python      # cluster-specific
source .venv/bin/activate

python exp4_train.py --config configs/exp4_default.yaml --data_path /path/to/dataset --save_models
