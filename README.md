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