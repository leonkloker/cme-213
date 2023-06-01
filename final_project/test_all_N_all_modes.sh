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

TAG=double

# Edit the lines below
TEST_DIR=$SLURM_SUBMIT_DIR/Outputs
# Must match the path
# string output_dir = "Outputs";
# in main.cpp

GPU_ARCHIVE_DIR=$SLURM_SUBMIT_DIR/Outputs_$TAG

# MAKEFILE_NAME=Makefile_$TAG
# make clean
# make -f ${MAKEFILE_NAME} -j

for mode in 1 2 3; do

  for N in 1 2 3 4; do 

    echo -e "\n\nmpirun -np ${N} ./main -d -g ${mode}"
    mpirun -np ${N} ./main -d -g ${mode}

    if [ ! -d ${GPU_ARCHIVE_DIR} ]; then
      mkdir ${GPU_ARCHIVE_DIR}
    fi
    cp ${TEST_DIR}/{CpuGpuDiff-${N}-${mode}.txt,NNErrors-${N}-${mode}.txt} ${GPU_ARCHIVE_DIR}/   
  done 
done

echo -e "\n*** Summary ***\n"

for mode in 1 2 3; do  
  for N in 1 2 3 4; do 
    tail -n 1 ${GPU_ARCHIVE_DIR}/CpuGpuDiff-${N}-${mode}.txt
  done 
done

echo -e "\n*** Grading mode 4 ***\n"

echo "main -g 4"
mpirun ./main -g 4

echo -e "\n*** Tests are complete ***"