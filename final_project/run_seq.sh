#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=CME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

cd $SLURM_SUBMIT_DIR

# Precision
TAG=double

# Edit the lines below
CPU_SAVE_DIR=/home/darve/fp_output
# This should match the path in main.cpp.

# Append _$TAG to the path
CPU_ARCHIVE_DIR=${CPU_SAVE_DIR}_$TAG

# MAKEFILE_NAME=Makefile_$TAG
# Make sure this Makefile exists or use the appropriate Makefile.

# make clean
# make -f ${MAKEFILE_NAME} -j

mpirun -np 4 ./main -s -d -n 32 -b 800 -l 0.001 -e 1 -p 5

for mode in 1 2 3; do
  echo Generating data for grading mode: ${mode}  
  mpirun -np 4 ./main -s -g ${mode}
done

if [ ! -d ${CPU_ARCHIVE_DIR} ]; then
  mkdir ${CPU_ARCHIVE_DIR}
fi
cp ${CPU_SAVE_DIR}/* ${CPU_ARCHIVE_DIR}/

echo -e "\n*** Data generation is complete ***"