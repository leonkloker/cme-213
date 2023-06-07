#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --partition=CME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

cd $SLURM_SUBMIT_DIR

ncu --set full -f -o ncompute ./main -g 4