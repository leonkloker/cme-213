#!/bin/bash
#SBATCH -o slurm.sh.out
#SBATCH -p CME
#SBATCH --gres gpu:1


### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo
make

echo
echo Output from main_q1
echo ----------------
./main_q1

echo
echo Output from main_q2
echo ----------------
./main_q2

echo
echo Output from main_q3
echo ----------------
./main_q3
