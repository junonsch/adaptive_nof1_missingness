#!/bin/bash  -eux
#SBATCH --job-name=missing_simulation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=juliana.schneider@hpi.de
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=4 # -c
#SBATCH --mem=32gb
#SBATCH --gpus=2
#SBATCH --time=8:00:00
#SBATCH --output=/dhc/home/juliana.schneider/adaptive_nof1/logs/%j.log # %j is job id

srun /dhc/home/juliana.schneider/conda3/envs/adaptive_nof1/bin/python ~/adaptive_nof1/notebooks/missing_simulation.py