#!/usr/bin/env python

###########################################################################
# Use STARE partitions to construct something resembling the original granule.
###########################################################################

import os, fnmatch

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

from dask.distributed import Client

import numpy as np

import geodata as gd
import h5py as h5
from pyhdf.SD import SD, SDC

from stopwatch import sw_timer as timer

import pystare as ps

from sortedcontainers import SortedDict

###########################################################################
#
def safe_shape(x):
    try:
        ret = x.shape
    except:
        ret = None
        pass
    return ret
#
###########################################################################
#
def slam(client,action,data,partition_factor=1.5):
    np = sum(client.nthreads().values())
    print('slam: np = %i'%np)
    shard_bounds = [int(i*len(data)/(1.0*partition_factor*np)) for i in range(int(partition_factor*np))] 
    if shard_bounds[-1] != len(data):
        shard_bounds = shard_bounds + [-1]
    data_shards = [data[shard_bounds[i]:shard_bounds[i+1]] for i in range(len(shard_bounds)-1)]
    print('ds len:        ',len(data_shards))
    print('ds item len:   ',len(data_shards[0]))
    print('ds type:       ',type(data_shards[0]))
    # print('ds dtype:      ',data_shards[0].dtype)
    big_future = client.scatter(data_shards)
    results    = client.map(action,big_future)
    return results
#
###########################################################################
#


# s302c800000000006.t007d5c6d54800092.MOD05_L2.A2005349.2120.061.2017294065852.sketchJ0.h5

def sare_from_fname(fname):
    return int(fname.split('.')[0][1:],16)
#    return int('0x'+fname.split('.')[0][1:],0)

# take a list of files, get their sare, and then group them according to their occupancy in a coarser level
def build_coarser_level(fnames):
    sares = [sare_from_fname(i) for i in fnames]
    resolutions = [gd.spatial_resolution(i) for i in sares]
    levels = set(resolutions)
    # print('levels: ',levels)
    if len(levels) != 1:
        raise ValueError('Filenames indicate multiple resolutions.')
    resolution = resolutions[0]
    coarser = {}
    for i in fnames:
        si = gd.spatial_clear_to_resolution(gd.spatial_coerce_resolution(sare_from_fname(i),resolution-1))
        if si not in coarser.keys():
            coarser[si] = []
        coarser[si].append(i)
    return coarser

def make_virtual(base_name,fnames,dataset_names,realign=False,sort=True):
    n_files = len(fnames)
    layouts = {}
    t0 = timer.current()
    for dsn in dataset_names:
        with h5.File(fnames[0],'r') as h:
            ds_dtype = h[dsn].dtype
        var_ns      = []
        var_n_cumul = [0]
        # print('fnames: ',fnames)
        for f in fnames:
            with h5.File(f,'r') as h:
                print(f,' ',dsn,' shape= ',h[dsn].shape)
                var_ns.append(h[dsn].shape[0]) # Assume 1D
                var_n_cumul.append(var_n_cumul[-1]+var_ns[-1])
        var_n_total = var_n_cumul[-1]
        print('  layout shape: ',var_n_total)
        layouts[dsn] = h5.VirtualLayout(shape=(var_n_total,),dtype=ds_dtype)
        # print(dsn,' dsn, n_tot ',var_n_total)
        # print(dsn,' dsn cum    ',var_n_cumul)
        verbose = True
        for i in range(n_files):
            t1 = timer.current()
            if var_ns[i] > 0:
                print(i,' vs shape: ',var_ns[i],' from file ',fnames[i])
                vs = h5.VirtualSource(fnames[i],dsn,shape=(var_ns[i],)) # could get shape from h
                with h5.File(fnames[i]) as h:
                    print('    100: ',dsn,' from ',fnames[i])
                    # print('100: ',h[dsn])
                    # print('101: ',h[dsn].dtype)
                    # print('102: ',h[dsn].dtype.names)
                    # print('')
                    if sort and 'src_coord' in h[dsn].dtype.names:
                        j    = 0
                        nseg = 0
                        segs = SortedDict()
                        while j < h[dsn]['src_coord'].shape[0]:
                            j0   = j
                            k0   = h[dsn]['src_coord'][j0]
                            k1   = k0 + 1
                            done = False
                            j1   = j + 1
                            while not done:
                                if j1 < h[dsn]['src_coord'].shape[0]:
                                    if k1 == h[dsn]['src_coord'][j1]:
                                        j1 = j1 + 1
                                        k1 = k1 + 1
                                    else:
                                        done = True
                                else:
                                    done = True
                            nseg = nseg + 1
                            segs[h[dsn]['src_coord'][j0]] = (j0,j1)
                            j = j1
                            if verbose:
                                print('  layout %s[%s] %5.2f%% complete, added %i points of %i in %s segments so far after %i seconds.'\
                                      %(fnames[i],dsn,int(100.0*j/var_ns[i]),j,var_ns[i],nseg,timer.current()-t1)
                                      ,end='\r',flush=True)
                            ## # print('layout k=%i,%i j=%i,%i'%(k0,k1,j0,j1))
                            ## if realign:
                            ##     layouts[dsn][k0:k1] = vs[j0:j1]
                            ## else:
                            ##     layouts[dsn][j0:j1] = vs[j0:j1]
                            ## j = j1
                        l = 0
                        for k in segs:
                            j0,j1 = segs[k]
                            if realign:
                                k0 = h[dsn]['src_coord'][j0]
                                k1 = h[dsn]['src_coord'][j1-1]
                            else:
                                k0 = l
                                k1 = l + j1-j0
                                l = l + j1-j0
                            print(k,' segs j= ',j0,j1,' k= ',k0,k1)
                            layouts[dsn][k0:k1] = vs[j0:j1]
                        print('   Added %i of %i items in %i segments in %i seconds from %s.'%(j,var_ns[i],nseg,timer.current()-t1,fnames[i]))
                    else:
                        print('  assigning ds of shape %s to layout %s.'%(vs.shape,dsn))
                        layouts[dsn][var_n_cumul[i]:var_n_cumul[i+1]] = vs[:]
            # else:
            #    raise ValueError('Size of source dataset %s unexpectedly greater than 1, equals %i.'%(dsn,var_ns[i]))
                    
    vds_name = '%s'%(base_name)
    # print('writing ',vds_name)
    with h5.File(vds_name,'w') as h:
        for i in layouts: # i.e. dsn's
            # print(' creating virtual dataset ',i)
            h.create_virtual_dataset(i,layouts[i])
    print('wrote %s in %i seconds.'%(vds_name,timer.current()-t0))
    return vds_name
#
###########################################################################
#

def make_virtual_from(inputs):
    ret=[]
    for vars in inputs:
        base_name,leaves,dataset_names,realign,sort = vars
        print('vars: ',base_name,leaves,dataset_names,realign,sort)
        ret.append(make_virtual(base_name,leaves,dataset_names,realign=realign,sort=sort))
    return ret
#
###########################################################################
#
def main():

    mod05_granules =['MOD05_L2.A2005349.2120.061.2017294065852']

    j0_resolution = 4
    filelist = os.listdir('./')
    files=[]
    for entry in filelist:
        # print(entry,'*'+mod05_granules[0]+'.sketchJ0.h5')
        if fnmatch.fnmatch(entry,'*'+mod05_granules[0]+'.sketchJ0.h5'):
            # if fnmatch.fnmatch(entry,'*06.t*J0.h5'):
            if fnmatch.fnmatch(entry,'*000%i.t*J0.h5'%j0_resolution):
                files.append(entry)

    # files=files[0:4]
    # for i in files:
    #    print(i,' i sare ',sare_from_fname(i))

    coarser = build_coarser_level(files)
    # print('coarser levels:   ',coarser)
    # print('coarser levels n: ',[len(coarser[i]) for i in coarser])

    # Serial
    if False:
        for i in coarser:
            base_name = 's%016x.%s.%s'%(i,'.'.join(mod05_granules[0].split('.')[:-2]),'sketchJ3.vds.h5')
            # print('base_name: ',base_name)
            sort_according_to_src_coord = False
            realign_output              = False
            make_virtual(\
                         base_name
                         ,coarser[i]
                         ,['metadata','wv_nir']
                         ,sort=sort_according_to_src_coord
                         ,realign=realign_output # Only enabled if sort is true
            )

    
    if True:
        #dbg
        client = Client()
        #dbg if False:
        #dbg     inputs = []
        #dbg     for i in coarser:
        #dbg         inputs.append(('s%016x.%s.%s'%(i,'.'.join(mod05_granules[0].split('.')[:-2]),'sketchJ3.vds.h5'),coarser[i],['metadata','wv_nir'],False,False))
        #dbg     # print('inputs: ',inputs)
        #dbg     # exit()
        #dbg     #dbg result = slam(client,make_virtual_from,inputs)
        #dbg     for i in inputs:
        #dbg         result = make_virtual_from(inputs)
        #dbg     #dbg client.gather(result)

        if True:
            done = False
            res = 3
            while not done:
                filelist = os.listdir('./')
                files=[]
                pattern = '*'+'.'.join(mod05_granules[0].split('.')[:-2])+'.sketchJ3.vds.h5'
                #print('pattern: ',pattern)
                for entry in filelist:
                    # print(entry,'*'+mod05_granules[0]+'.sketchJ0.h5')
                    if fnmatch.fnmatch(entry,pattern):
                        # if fnmatch.fnmatch(entry,'*0%i.t*J3.h5'%res):
                        match_pattern = '*0%i.M*J3.vds.h5'%res
                        #print('match: ',match_pattern)
                        if fnmatch.fnmatch(entry,match_pattern):
                            files.append(entry)
                #####
                #print('files: ',files)
                #exit()
                coarser = build_coarser_level(files)
                inputs = []
                for i in coarser:
                    inputs.append(('s%016x.%s.%s'%(i,'.'.join(mod05_granules[0].split('.')[:-2]),'sketchJ3.vds.h5'),coarser[i],['metadata','wv_nir'],False,False))
                # print('inputs: ',inputs)
                # exit()
                for i in inputs:
                    result = make_virtual_from(inputs)
                #dbg
                result = slam(client,make_virtual_from,inputs,partition_factor=1)
                #dbg
                client.gather(result)
                res = res - 1
                done = res < 2
            print(result)
            
        if True:
            result = make_virtual_from([('MOD05_L2.A2005349.2120.sketchJ3.vds.h5',['s2c00000000000001.MOD05_L2.A2005349.2120.sketchJ3.vds.h5','s3000000000000001.MOD05_L2.A2005349.2120.sketchJ3.vds.h5'],['metadata','wv_nir'],True,True)])
        print(result)
        
        #dbg
        client.close()
        
    return 


if __name__ == '__main__':
    main()
