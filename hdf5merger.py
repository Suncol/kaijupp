# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:07:41 2021

@author: sun
"""

import os 
import sys
import time
import numpy as np
import h5py
import re

# for performance tuning, not essential (you can comment the decorator by your own)
from numba import jit # it is a jit compiler, use it when the func call is in great amount
from joblib import Parallel, delayed # use the joblib, which is much quicker that multiprocessing in the python module

# for user interface / collecting user define vars
import argparse

# get all the data file to a file list
def get_file_list(path,basename):
    print ('Processing dir:' + path)
    print ('Load msphere data files without Res files')
    # delete Res files in output dir
    file_list = sorted(os.listdir(path))
    data_file_list = [file_name for file_name in file_list] # no just give a reference
    for file_name in file_list:
        if ("Res" in file_name) or not (basename in file_name) or not ('.h5' in file_name): # remove if not a useful h5 file data
            data_file_list.remove(file_name)
    try:
        data_file_list.remove(basename + '.mix.h5')
        data_file_list.remove(basename + '.volt.h5')
    except:
        print ("Can't find file: " + basename + '.mix.h5 or ' + basename + '.volt.h5\n')
        print  ("Maybe you are using omega model output file")
    return data_file_list

# get the absolute path 
def get_ap(filelist,path):
    data_path_list = []
    for filename in filelist:
        data_path_list.append(os.path.join(path, filename))
    return data_path_list

# check the number of the data files
def check_files(data_file_list,model_type):
    strs = data_file_list[-1]
    strs = strs[0:-3]
    strs = re.sub("[a-zA-Z]","",strs)

    # replace _ by space and get all the numbers
    strs = strs.split('_')
    strs = strs[1:] #remove the first "_"
    if model_type == 'omega':
        nums = list(map(int,strs))
        NI_rank = nums[0]
        NJ_rank = nums[1]
        NK_rank = nums[2]
        N_rank = nums[-1]
        if N_rank != NI_rank * NJ_rank - 1:
            print('N_rank should equal to NI*NJ, something goes wrong')
            sys.exit(0) # exit program
        else:
            print('Check file list ok, begin to merge hdf5 files')

    elif model_type == 'kaiju':
        strs = strs[:-1]
        nums = list(map(int,strs))
        NI_rank = nums[0]
        NJ_rank = nums[1]
        NK_rank = nums[2]
        len_file_list = len(data_file_list)
        if len_file_list != NI_rank * NJ_rank * NK_rank:
            print('N_rank should equal to NI*NJ, something goes wrong')
            sys.exit(0)
        else:
            print('Check file list ok, begin to merge hdf5 files')

# count the steps in the data files
def count_steps(fname):
    with h5py.File(fname,'r') as hf:
        grps = hf.values()
        grpNames = [str(grp.name) for grp in grps]
        #Steps = [stp if "/Step#" in stp for stp in grpNames]
        Steps = [stp for stp in grpNames if "/Step#" in stp]
        nSteps = len(Steps)
        step_list = []
        for i in range(nSteps):
            # step_list.append(Steps[i][6:nSteps-1])
            step_list.append(Steps[i][6:])
        # print(Steps[i])
        step_list = list(map(np.int,step_list)) # for python3 we need to convert it
        min_step = np.min(step_list)
        max_step = np.max(step_list)
        print('min_step: '+str(min_step)+'   max_step: '+str(max_step))
        return min_step,max_step

def get_dims(filename):
    with h5py.File(filename,'r') as hf:
        dim_d = np.shape(hf.get("X"))
    return (dim_d)

def get_dims_files(filelist,model_type):   
    '''
    the basename is detect, but you better write it on your own
    '''
    nfiles = len(filelist)
    ndim = np.size(get_dims(filelist[0]))
    # print('Data file numbers: ' + str(nfiles) + ';   dims: ' + str(ndim) )
    fdims_d = np.zeros((nfiles,ndim),dtype=np.int32)
    for idx, filename in enumerate(filelist):
        base, ext =  os.path.splitext(os.path.basename(filename))
        name_parts = base.split('_')
        if model_type == 'kaiju':
            name_parts[-1] = name_parts[-1].split('.')[0]
        pidx = np.array(name_parts[1:7], dtype=np.int32)
        # Fortran and C has reverse ordering
        fdims_d[idx,:] = np.flipud(pidx[3:6])
    return(name_parts[0],np.flipud(pidx[0:3]), fdims_d)

# get variable names in data file, just do the check in min step
def get_varnames(filename,min_step):
    step_name = '/Step#'+str(min_step)
    with h5py.File(filename,'r') as hf:
        varnames = list(hf[step_name].keys()) # mind that diff between py 2 and 3
        #varnames = hf.visit(['/Step#0'])
    return(varnames)

# write the xyz grid to the output merged files
def write_XYZ(fdict,wh5file):
    write_phase_dataset('/X', fdict,wh5file,True)
    if fdict['ndim'] > 1:
        write_phase_dataset('/Y', fdict,wh5file,True)
    if fdict['ndim'] > 2:
        write_phase_dataset('/Z', fdict,wh5file,True)
    return(0)

# write all data set, if the data is corner data, usePhase should be set to True
def write_phase_dataset(datname, fdict, wh5file,usePhase=False):
    h5dim = fdict['nXyz_d']
    if usePhase:
        h5dim = h5dim + 1
    h5data = wh5file.create_dataset(datname,
                                    h5dim,
                                    np.float32)
    for ifile, fname in enumerate(file_dict['readfiles']):
        with h5py.File(fname,'r') as hf:
            mydata = hf[datname]
            ib,ie, iib, iie = get_phase_idx(0,ifile,fdict,usePhase)
            if fdict['ndim'] == 2:
                jb,je,jjb,jje = get_phase_idx(1,ifile,fdict,usePhase)
                h5data[ib:ie,jb:je] = mydata[iib:iie,jjb:jje]
            elif fdict['ndim'] == 3:
                jb,je,jjb,jje = get_phase_idx(1,ifile,fdict,usePhase)
                kb,ke,kkb,kke = get_phase_idx(2,ifile,fdict,usePhase)
                h5data[ib:ie,jb:je,kb:ke] = mydata[iib:iie,jjb:jje,kkb:kke]
            else:
                h5data[ib:ie] = mydata[iib:iie]
    return(0)

# get the global index and local index (pysical cells)
def get_phase_idx(ijkDir,ifile,fdict,usePhase = False):
    # cell pahses are overlapping between blocks
    # therefore only add the extra phase to the last 
    # cell in the direction ijkDir
    the_end=0
    if(usePhase and fdict['nProc_d'][ijkDir] == fdict['ijkProc_d'][ifile,ijkDir] + 1 ):
        the_end = 1
    # indexes for the global matrix
    istart = fdict['ijkProc_d'][ifile,ijkDir]*fdict['nMyXyz_d'][ijkDir]
    iend  = istart + fdict['nMyXyz_d'][ijkDir] + the_end
    # indexes without ghost cells
    iistart = file_dict['ghost_d'][ijkDir]
    iiend   = iistart + fdict['nMyXyz_d'][ijkDir] + the_end
    #print "indexes :: ", istart,iend, iistart, iiend
    return(istart,iend, iistart, iiend)

# master func to write data (not grid data) to merged files
def write_step(fdict,it,wh5file):
    gid = "/Step#%d/" %(it)
    group = wh5file.create_group(gid)
    with h5py.File(fdict['readfiles'][0],'r') as hf:
        for attr in hf[gid].attrs.keys():
            group.attrs.create(attr,hf[gid].attrs[attr])

    for var in fdict['varnames']:
        gid = "Step#%d/"%(it) + var
        write_phase_dataset(gid, fdict,wh5file)
        #print "writing : ", gid
    return(gid)

# get the original merged files, not do the corner2center things
def write_merged_file(input_list):
    output_dir,basename,it,fdict = input_list
    print ("Merging output step#: %010d"%(it))
    outh5file = h5py.File(os.path.join(output_dir,basename+"Step#%010d" %(it)+".h5"),"w") # for most cases 10 digits is enough
    write_XYZ(fdict,outh5file)
    write_step(fdict,it,outh5file)
    outh5file.close()

### funcs for merge the subset of the data ###

# change from the volume corner to volume center
@jit(nopython=True, parallel=True)
def center_3d(v):
    return 0.125*(v[:-1, :-1, :-1]+v[1:, :-1, :-1]+v[:-1, 1:, :-1]+v[:-1, :-1, 1:] \
            +v[1:, 1:, :-1]+v[1:, :-1, 1:]+v[:-1, 1:, 1:]+v[1:, 1:, 1:])

# grid slice in the eq plane
@jit(nopython=True, parallel=True)
def grid_slice_eq(xCenter, yCenter):
    xCenter, yCenter = xCenter.T, yCenter.T
    ni, nj, nk = xCenter.shape # get the shape of the grid
    xe = np.zeros((ni,nj*2+1)) # grid shape of the joint plane 
    ye = np.zeros_like(xe)  

    # for i in range(ni):
    #     for j in range(nj):
    #         xe[i,j] = xCenter[i,j,0]
    #         ye[i,j] = yCenter[i,j,0]

    #     for j in range(nj,2*nj):
    #         xe[i,j] = xCenter[i,2*nj-1-j,nk//2-1]
    #         ye[i,j] = yCenter[i,2*nj-1-j,nk//2-1]
    
    xe[:,:nj] = xCenter[:,:,0]
    ye[:,:nj] = yCenter[:,:,0]
        
    xe[:,nj:-1] = xCenter[:,::-1,nk//2-1]
    ye[:,nj:-1] = yCenter[:,::-1,nk//2-1]

    xe[:,-1] = xe[:,0]
    ye[:,-1] = ye[:,0]

    return xe, ye

# grid slice in the xz plane
@jit(nopython=True, parallel=True)
def grid_slice_xz(xCenter, zCenter):
    xCenter, zCenter = xCenter.T, zCenter.T
    ni, nj, nk = xCenter.shape # get the shape of the grid
    xe = np.zeros((ni,nj*2+1)) # grid shape of the joint plane 
    ye = np.zeros_like(xe)  

    # for i in range(ni):
    #     for j in range(nj):
    #         xe[i,j] = xCenter[i,j,nk//4]
    #         ye[i,j] = zCenter[i,j,nk//4]
        # for j in range(nj,nj*2):
        #     xe[i,j] = xCenter[i,2*nj-1-j,nk//4*3]
        #     ye[i,j] = zCenter[i,2*nj-1-j,nk//4*3]
    
    xe[:,:nj] = xCenter[:,:,nk//4]
    ye[:,:nj] = zCenter[:,:,nk//4]
        
    xe[:,nj:-1] = xCenter[:,::-1,nk//4*3]
    ye[:,nj:-1] = zCenter[:,::-1,nk//4*3]


    xe[:,-1] = xe[:,0]
    ye[:,-1] = ye[:,0]

    return xe, ye

# write the xy or xz plane grid, so the origin grid should be 3D 
def write_XYZ_slice(fdict,wh5file,slice_type):
    if fdict['ndim'] < 3:
        raise RuntimeError('not a 3D grid!')
    h5dim = fdict['nXyz_d']
    h5dim_ = np.zeros((2,), dtype=np.int32)

    h5dim_[0] = h5dim[2] # i index 
    h5dim_[1] = h5dim[1]*2+1 # j index

    if slice_type == 'eq':
        # create dataset of grids
        h5datax = wh5file.create_dataset('/X',
                                    h5dim_,
                                    np.float32)
        h5datay = wh5file.create_dataset('/Y',
                                    h5dim_,
                                    np.float32)

        gridx = np.zeros(h5dim+1)
        gridy = np.zeros(h5dim+1)
        gridxc = np.zeros(h5dim)
        gridyc = np.zeros(h5dim)

        for ifile, fname in enumerate(file_dict['readfiles']):
            with h5py.File(fname,'r') as hf:
                ib,ie, iib, iie = get_phase_idx(0,ifile,fdict,True)
                jb,je,jjb,jje = get_phase_idx(1,ifile,fdict,True)
                kb,ke,kkb,kke = get_phase_idx(2,ifile,fdict,True)
                
                datax = hf['/X']
                datay = hf['/Y']

                gridx[ib:ie,jb:je,kb:ke] = datax[iib:iie,jjb:jje,kkb:kke]
                gridy[ib:ie,jb:je,kb:ke] = datay[iib:iie,jjb:jje,kkb:kke]

        gridxc = center_3d(gridx)
        gridyc = center_3d(gridy)
        h5datax[:,:],h5datay[:,:] = grid_slice_eq(gridxc,gridyc)
    
    if slice_type == 'xz':
      
        # create dataset of grids
        h5datax = wh5file.create_dataset('/X',
                                    h5dim_,
                                    np.float32)
        h5dataz = wh5file.create_dataset('/Z',
                                    h5dim_,
                                    np.float32)

        gridx = np.zeros(h5dim+1)
        gridz = np.zeros(h5dim+1)
        gridxc = np.zeros(h5dim)
        gridzc = np.zeros(h5dim)

        for ifile, fname in enumerate(file_dict['readfiles']):
            with h5py.File(fname,'r') as hf:
                ib,ie, iib, iie = get_phase_idx(0,ifile,fdict,True)
                jb,je,jjb,jje = get_phase_idx(1,ifile,fdict,True)
                kb,ke,kkb,kke = get_phase_idx(2,ifile,fdict,True)
                
                datax = hf['/X']
                dataz = hf['/Z']

                gridx[ib:ie,jb:je,kb:ke] = datax[iib:iie,jjb:jje,kkb:kke]
                gridz[ib:ie,jb:je,kb:ke] = dataz[iib:iie,jjb:jje,kkb:kke]

        gridxc = center_3d(gridx)
        gridzc = center_3d(gridz)

        h5datax[:,:],h5dataz[:,:] = grid_slice_xz(gridxc,gridzc)
        
    return(0)

# slice the data in the eq plane
@jit(nopython=True, parallel=True)
def data_slice_eq(value):
    value = value.T
    ni, nj, nk = value.shape # get the shape of the grid
    data = np.zeros((ni,nj*2+1)) # grid shape of the joint plane 
  
    # for i in range(ni):
    #     for j in range(nj):
    #         data[i,j] = (value[i,j,0]+value[i,j,-1]) / 2 

    #     for j in range(nj,2*nj):
    #         data[i,j] = (value[i,2*nj-1-j,0]+value[i,2*nj-1-j,-1]) / 2

    data[:,:nj] = value[:,:,0]
    data[:,nj:-1] = value[:,::-1,nk//2-1]

    data[:,-1] = data[:,0]

    return data

# slice the data in the xz plane
@jit(nopython=True, parallel=True)
def data_slice_xz(value):
    value = value.T
    ni, nj, nk = value.shape # get the shape of the grid
    data = np.zeros((ni,nj*2+1)) # grid shape of the joint plane 
    
    # for i in range(ni):
    #     for j in range(nj):
    #         data[i,j] = value[i,j,nk//4] 

    #     for j in range(nj,2*nj):
    #         data[i,j] = value[i,2*nj-1-j,nk//4*3]

    data[:,:nj] = value[:,:,nk//4]
    data[:,nj:-1] = value[:,::-1,nk//4*3]

    data[:,-1] = data[:,0]

    return data

# write the slice plane to the merged files
def write_data_slice(dataname,fdict,wh5file,slice_type):
    if fdict['ndim'] < 3:
        raise RuntimeError('not a 3D grid!')
    h5dim = fdict['nXyz_d']
    h5dim_ = np.zeros((2,), dtype=np.int32)

    h5dim_[0] = h5dim[2]
    h5dim_[1] = h5dim[1]*2+1

    if slice_type == 'eq':
        # create dataset of grids
        h5data = wh5file.create_dataset(dataname,
                                    h5dim_,
                                    np.float32)

        datam = np.zeros(h5dim)

        for ifile, fname in enumerate(file_dict['readfiles']):
            with h5py.File(fname,'r') as hf:
                ib,ie, iib, iie = get_phase_idx(0,ifile,fdict,True)
                jb,je,jjb,jje = get_phase_idx(1,ifile,fdict,True)
                kb,ke,kkb,kke = get_phase_idx(2,ifile,fdict,True)
                
                data = hf[dataname]

                datam[ib:ie,jb:je,kb:ke] = data[iib:iie,jjb:jje,kkb:kke]

        h5data[:,:] = data_slice_eq(datam)
    
    if slice_type == 'xz':
        # create dataset of grids
        h5data = wh5file.create_dataset(dataname,
                                    h5dim_,
                                    np.float32)

        datam = np.zeros(h5dim)

        for ifile, fname in enumerate(file_dict['readfiles']):
            with h5py.File(fname,'r') as hf:
                ib,ie, iib, iie = get_phase_idx(0,ifile,fdict,True)
                jb,je,jjb,jje = get_phase_idx(1,ifile,fdict,True)
                kb,ke,kkb,kke = get_phase_idx(2,ifile,fdict,True)
                
                data = hf[dataname]

                datam[ib:ie,jb:je,kb:ke] = data[iib:iie,jjb:jje,kkb:kke]
        
        h5data[:,:] = data_slice_xz(datam)

    return(0)

# master func to write data (not grid data) to the merged files
def write_slice_dataset(fdict,it,wh5file,slice_type):
    gid = "/Step#%d/" %(it)
    group = wh5file.create_group(gid) # create a group name with the gid/step
    with h5py.File(fdict['readfiles'][0],'r') as hf:
        for attr in hf[gid].attrs.keys():
            group.attrs.create(attr,hf[gid].attrs[attr])
    
    for var in fdict['varnames']:
        gid = "Step#%d/"%(it) + var 
        write_data_slice(gid,fdict,wh5file,slice_type)
    return(gid)

# master func of write slice plane data and grids
def write_step_slice(input_list):
    output_dir, basename, it, fdict, slice_type = input_list
    print ("Merging output step#: %010d"%(it))
    outh5file = h5py.File(os.path.join(output_dir,basename+'_'+slice_type+"_Step#%010d" %(it)+".h5"),"w") # for most cases 10 digits is enough
    write_XYZ_slice(fdict,outh5file,slice_type)
    write_slice_dataset(fdict,it,outh5file,slice_type)
    #write_step(fdict,it,outh5file)
    outh5file.close()


# main func, user should set his/her own setting here
if __name__ == "__main__":
    # set default settings
    hfdType = np.float32
    sys.stdout.flush()
    
    ###------
    # # user define variables, you can modify the vars here if you don't wanna use the pbs/slurm manager
    # hdf5_fldr = '/glade/scratch/luanxl/model_result/kaiju/test/earth_test'
    # basename = 'msphere' 
    # model_type = 'kaiju'
    # slice_type = 'eq'
    # do_origin_merge = 0 # just merge slice dataset, not all of the original dataset, because it's really lag
    # save_step = 1 # step to save the merged files
    # n_workers = 36 # numbers of the workers in the joblib, better not larger than cpu cores 
    #                # if the n_workers == 0, we will just use the non-parallel version, a simple loop
    
    ###------

    ###------
    # user define vars, collecting from the argparse and set some default configs
    parser = argparse.ArgumentParser(description='HDF5 files merger for kaiju and omega!')
    parser.add_argument('--hdf5_fldr', '-hf', help='path to the h5 files', required=True)
    parser.add_argument('--basename','-bn',help='basename of the model results / Runid', required=True)
    parser.add_argument('--model_type','-t',help='model type, kaiju or omega',default='kaiju')
    parser.add_argument('--slice_type','-st',help='slice type',required=True)
    parser.add_argument('--do_origin_merge','-om',help='if do the original merge,defalut is false',default=0)
    parser.add_argument('--save_step','-s',help='save step',default=1)
    parser.add_argument('--n_workers','-nw',help='number of workers in parallel merging',default=5)
    args = parser.parse_args()

    hdf5_fldr = args.hdf5_fldr
    basename = args.basename
    model_type = args.model_type
    slice_type = args.slice_type
    do_origin_merge = int(args.do_origin_merge)
    save_step = int(args.save_step)
    n_workers = int(args.n_workers)

    ###------

    output_dir = os.path.join(hdf5_fldr, 'merged_files')
    
    # if the output_dir is not found, build it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    h5file_list = get_file_list(hdf5_fldr, basename)
    # print (h5file_list) # sometime print it is meaningless
    
    check_files(h5file_list, model_type) # check the file lists

    h5file_list = get_ap(h5file_list, hdf5_fldr) # get the absolute path, use this script anywhere
    
    [min_step, max_step] = count_steps(h5file_list[0]) # get the numbers of the steps
    
    ntimes = max_step - min_step + 1
    ghost_d = np.array([0,0,0])
    dims_d = get_dims(h5file_list[0])

    basename, nProc_d, ijkProc_d = get_dims_files(h5file_list,model_type) 
    ndim = ijkProc_d.shape[1]
    # print (ijkProc_d)

    # get number of cell centers
    nMyXyz_d = np.zeros(ndim,dtype=np.int32)
    nMyXyz_d[0] = (dims_d[0] - 1 - 2*ghost_d[0])
    nMyXyz_d[1] = (dims_d[1] - 1 - 2*ghost_d[1])
    nMyXyz_d[2] = (dims_d[2] - 1 - 2*ghost_d[2])

    nXyz_d = np.zeros(ndim,dtype=np.int32)
    nXyz_d[0] = nProc_d[0]*nMyXyz_d[0]
    nXyz_d[1] = nProc_d[1]*nMyXyz_d[1]
    nXyz_d[2] = nProc_d[2]*nMyXyz_d[2]

    file_dict = {} # the info of the model results
    file_dict['nXyz_d']    = nXyz_d
    file_dict['nProc_d']   = np.array(nProc_d,dtype=np.int32)
    file_dict['ijkProc_d'] = np.array(ijkProc_d, dtype=np.int32)
    file_dict['nMyXyz_d']  = nMyXyz_d
    file_dict['ntimes']    = ntimes
    file_dict['ghost_d']   = np.array(ghost_d, dtype=np.int32)
    file_dict['readfiles'] = h5file_list
    file_dict['ndim']      = ndim
    file_dict['varnames']  = get_varnames(h5file_list[0],min_step)

    
    # # simple loop for merge the original data set (no plane slice and corner2center trans)
    # only for unit test
    # for it in range(min_step,max_step+1,save_step): # merge data, one step in one file
    #     write_merged_file([output_dir,basename,it,file_dict])
    #     print ("Merged!")

    # do the simple loop for merging the original data setl, mind that it can be really slow
    if do_origin_merge == 1:
        print('Do the original dataset merging!')
        for it in range (min_step,max_step+1,save_step): 
            time_start = time.time()
            write_merged_file([output_dir,basename,it,file_dict])
            time_end = time.time()
            print('Time cost ', time_end-time_start,'s')
            print("Merged!")
            print('\n')
    
    # merge the slice plane, using simple loop or parallel processes
    if n_workers == 0:
        # simple loop for merge the slice data set (plane slice and corner2center trans)
        print('merging grid in the ' + slice_type + ' plane!')
        for it in range(min_step,max_step+1,save_step): # merge data, one step in one file
            time_start = time.time() # record the time that the slice merge begin
            write_step_slice([output_dir,basename,it,file_dict,slice_type])
            time_end = time.time() # record the time that the slice merge end
            print('Time cost ', time_end-time_start, 's')
            print ("Merged!")
            print ('\n')

    elif n_workers > 0:
        # use the joblib to parallel merge process between steps, but you'd better submit as a job
        # no need for timing
        # highly recommend user to use this, if the io bound is not the bottleneck, that can be really quick
        args_list = [[output_dir,basename,it,file_dict,slice_type] for it in range(min_step,max_step,save_step)] 
        Parallel(n_jobs=n_workers,verbose=100)(delayed(write_step_slice) (args) \
                for args in args_list)
    

 
    