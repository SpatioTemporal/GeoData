#!/usr/bin/env python

# <<<<<<< HEAD

###########################################################################
# Use STARE partitions to construct something resembling the original granule.
###########################################################################

import os, fnmatch

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

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

def main():

    mod05_granules = [\
                      'MOD05_L2.A2005349.2115.061'
                      ,'MOD05_L2.A2005349.2120.061'
                      ,'MOD05_L2.A2005349.2125.061'
                      ,'MOD05_L2.A2005349.2130.061'
                      ]

    mod05_granules = [\
                      'MOD05_L2.A2005349.2120.061'
                      ]
    
    filelist = os.listdir('./')
    files=[]
    for entry in filelist:
        if fnmatch.fnmatch(entry,'*'+mod05_granules[0]+'*.sketchJ0.h5'):
            files.append(entry)
    print('files: ',files)
    spart_names = files
    print('len spart_names: ',len(spart_names))
    # exit()

    var_ns = []
    var_n_cumul = [0]
    for sname in spart_names:
        with h5.File(sname,'r') as h:
            # Define the layout of the virtual data set.
            var_ns.append( h['wv_nir'].shape[0] )
            var_n_cumul.append(var_n_cumul[-1]+var_ns[-1])
    var_n_total = var_n_cumul[-1]

    print('')
    print('var_ns:      ',var_ns)
    print('var_n_cumul: ',var_n_cumul)
    print('var_n_total: ',var_n_total)

    with h5.File(spart_names[0],'r') as h:
        s5_vars_dtype     = h['wv_nir'].dtype
        s5_metadata_dtype = h['metadata'].dtype
        
    layout    = h5.VirtualLayout(shape=(var_n_total,),dtype=s5_vars_dtype)
    print('layout:      ',layout)
        
    layout_md = h5.VirtualLayout(shape=(len(spart_names),),dtype=s5_metadata_dtype)
    print('layout_md:   ',layout_md)
    print('')

    layout_idx = np.arange(var_n_total)

    if False:
        for i in range(len(spart_names)):
            print(i,' part: ',var_n_cumul[i],var_n_cumul[i+1],', ',var_ns[i])
        print('')
        
    # print('s5_metadata_dtype: ',s5_metadata_dtype)
    ds_name = 'wv_nir'
    t0 = timer.current()
    t00 = t0
    # for i in range(3):
    # for i in range(6):
    for i in range(len(spart_names)):
        timer.stamp('main0')
        t1 = timer.current()
        print('laying out %s with %i points. Completed %i so far, %5.2f%%, with %i s total elapsed.'%(spart_names[i],var_ns[i],var_n_cumul[i],100*var_n_cumul[i]/var_n_total,t1-t00))
        # Just concatenate
        #++ layout[var_n_cumul[i]:var_n_cumul[i+1]] = h5.VirtualSource(spart_names[i],ds_name,shape=(var_ns[i],))
        # What I would like to do...
        if var_ns[i] != 0:
            with h5.File(spart_names[i]) as h:
                vs = h5.VirtualSource(spart_names[i],ds_name,shape=(var_ns[i],))
                # print('vs: ',vs)
                j = 0
                nseg = 0
                while j < h[ds_name]['src_coord'].shape[0]:
                    j0 = j
                    k0 = h[ds_name]['src_coord'][j0]
                    k1 = k0+1
                    done = False
                    j1 = j+1
                    while not done:
                        if j1 % 100 == 0:
                            t2 = timer.current()                            
                            print('layout %2d%% complete, added %i items in %i segments, %i s elapsed.'%(int(100.0*j1/var_ns[i]),j,nseg,t2-t1),end='\r',flush=True)
                        if j1 < h[ds_name]['src_coord'].shape[0]:
                            if k1 == h[ds_name]['src_coord'][j1]:
                                j1 = j1+1
                                k1 = k1+1
                            else:
                                done = True
                        else:
                            done = True
                    # print('k0,k1: ',k0,k1)
                    # print('100: ',h[ds_name]['src_coord'][k0:k1])
                    # print('101: ',vs[k0:k1])
                    # print('102: ',)
                    nseg = nseg + 1
                    layout[k0:k1] = vs[j0:j1]
                    j = j1
                print('   Added %i of %i items in %i segments in %i seconds from %s.'%(j,var_ns[i],nseg,timer.current()-t1,spart_names[i]))
        timer.stamp('main1')
        t0 = t1
        # else: # zero-case
        #    layout[var_n_cumul[i]:var_n_cumul[i+1]] = h5.VirtualSource(spart_names[i],ds_name,shape=(var_ns[i],))                        
        
    for i in range(len(spart_names)):
        layout_md[i] = h5.VirtualSource(spart_names[i],'metadata',shape=(1,))

    tmp_data_src_name = '.'.join(spart_names[0].split('.')[2:-2]) # Make tmp_data_src_name from spart_names common base
    vds_fname = '.'.join([tmp_data_src_name,ds_name,'sketchJ0.vds.h5'])
    print('writing ',vds_fname)
    with h5.File(vds_fname,'w',libver='latest') as f:
        f.create_virtual_dataset('wv_nir',layout)      # fillvalue?
        f.create_virtual_dataset('metadata',layout_md) # fillvalue?
        # f.create_dataset(['image_lookup']...)

    print('MODIS Sketching Done')
    return

# =======
if __name__ == '__main__':
    main()
