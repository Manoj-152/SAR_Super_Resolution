#!/bin/bash
#SBATCH --job-name=DL-Proj-SRUN2
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1 --tasks-per-node=1 --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=90GB
#SBATCH --output=DL-Proj-2.log
#SBATCH --error=DL-Proj_err-2.log
module purge
bash run_2.sh