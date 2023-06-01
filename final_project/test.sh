#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --partition=CME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

echo "Starting at `date`"
echo "Current working directory is $SLURM_SUBMIT_DIR"
echo "The master node of this job is: $SLURMD_NODENAME"
echo "Number of compute nodes: $SLURM_JOB_NUM_NODES"
echo "Compute node names: $SLURM_JOB_NODELIST"
echo "Using $SLURM_TASKS_PER_NODE tasks per node."
echo "Number of CPUs on node: $SLURM_CPUS_ON_NODE."
echo ----------------

cd $SLURM_SUBMIT_DIR

# Select the number of MPI processes to use
N=4

# This script will run all 4 modes with your choice of N

# Runs mode 1, 2, and 3
for mode in 1 2 3; do
  echo -e "\n* Mode ${mode} *"
  echo mpirun -np ${N} ./main -g ${mode}  
  mpirun -np ${N} ./main -g ${mode}
done

echo -e "\n*** Summary ***\n"

for mode in 1 2 3; do
  tail -n 1 Outputs/CpuGpuDiff-${N}-${mode}.txt
done

echo -e "\n*** Grading mode 4 ***\n"

echo "main -g 4"
mpirun ./main -g 4

echo -e "\n*** Tests are complete ***"