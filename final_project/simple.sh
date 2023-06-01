#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --partition CME
#SBATCH --qos cme
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

mpirun -n 1 ./main -d -n 32 -b 800 -l 0.001 -e 1 -p 5