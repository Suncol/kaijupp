# Postprocessing python lib for kaiju (a MHD model)

## Install dependency python package 
Highly suggest your to build up a visual python env, using conda or pip. Here we take pip as an example.
- First create the python env with venv
- Then get into the visual env and run this command:
```
python -m pip install -r requirements.txt
```
- Everything finished!

## intro for merge scripts
### simple loop version
Just a simple loop all files and merge the grid and data with the correct order. The code is optimized by numba jit.

### parallel version
Use joblib to parallel the merge process between different step, user can assign the number of the workers to execute the script, usually it is equal to the number of the cpu cores that the machine has. This version is much faster that the simple loop version.

## usage
### Run directly
Just for lightly use, run the python script right on the login node with command line input. The usage can be found here:
```
[env] cheyenne4 earth_test/kaijupp> python hdf5merger.py -h
usage: hdf5merger.py [-h] --hdf5_fldr HDF5_FLDR --basename BASENAME
                     [--model_type MODEL_TYPE] --slice_type SLICE_TYPE
                     [--do_origin_merge DO_ORIGIN_MERGE]
                     [--save_step SAVE_STEP] [--n_workers N_WORKERS]

HDF5 files merger for kaiju and omega!

optional arguments:
  -h, --help            show this help message and exit
  --hdf5_fldr HDF5_FLDR, -hf HDF5_FLDR
                        path to the h5 files
  --basename BASENAME, -bn BASENAME
                        basename of the model results / Runid
  --model_type MODEL_TYPE, -t MODEL_TYPE
                        model type, kaiju or omega
  --slice_type SLICE_TYPE, -st SLICE_TYPE
                        slice type
  --do_origin_merge DO_ORIGIN_MERGE, -om DO_ORIGIN_MERGE
                        if do the original merge,defalut is false
  --save_step SAVE_STEP, -s SAVE_STEP
                        save step
  --n_workers N_WORKERS, -nw N_WORKERS
                        number of workers in parallel merging
```

### Run using the job manager
Use the script in the places folder.