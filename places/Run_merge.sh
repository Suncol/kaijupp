#!/bin/bash 

#### the script is for tianhe-2 ####

# just use a node is ok
#SBATCH -N 1 -n 1 


# user define options
hdf5_flder=/glade/scratch/luanxl/model_result/kaiju/test/earth_test
basename=msphere
slice_type='xz'

source ~/.kaiju_env # mind to load the anaconda3 env

# run the merge file
yhrun -N $SLURM_NNODES -n $SLURM_NPROCS -c 24 ../hdf5merger.py -hf ${hdf5_flder} -bn ${basename} -st ${slice_type} > merged.out
