#!/bin/bash 

#### the script is for tianhe-2 ####

# just use a node is ok
#SBATCH -N 1 -n 1 

# run the merge file
yhrun -N $SLURM_NNODES -n $SLURM_NPROCS -c 24 ../hdf5merger.py
