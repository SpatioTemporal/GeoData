
#!/usr/bin/env python

import os

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

vds_files =\
    ['MOD05_L2.A2005349.2115.061.2017294065841.wv_nir.vds.h5'
     ,'MOD05_L2.A2005349.2120.061.2017294065852.wv_nir.vds.h5'
     ,'MOD05_L2.A2005349.2125.061.2017294065400.wv_nir.vds.h5'
     ,'MOD05_L2.A2005349.2130.061.2017294065345.wv_nir.vds.h5']

if False:
    for i in vds_files:
        with h5.File(i,'r') as f:
            print(\
                  '"%s"=file, shape=%s, n_max=%i'
                  %(i,f['metadata'].shape,np.amax(f['metadata']['n_data'].reshape(f['metadata'].shape[0])))
                  ,', data shape=%s'%(f['wv_nir'].shape,))

###########################################################################
proj=ccrs.PlateCarree()
transf = ccrs.Geodetic()

def init_figure(proj):
    plt.figure()
    ax = plt.axes(projection=proj)
    # ax.set_global()
    # ax.coastlines()
    return ax

ax = init_figure(proj)
i = vds_files[0]
if True:
    with h5.File(i,'r') as f:
        sh = f['wv_nir'].shape
        for j in range(sh[0]):
        # for j in range(30):
            sare = f['wv_nir']['sare'][j,:]
            idx  = np.where(sare>0)
            # print(i,j,sare.shape)
            # print(idx[0])
            wv   = f['wv_nir']['Water_Vapor_Near_Infrared'][j,:]
            print(np.amin(wv),np.amax(wv))
            vmin = -10
            vmax = 0
            if len(idx[0]) > 0:
                lat,lon = ps.to_latlon(sare[idx])
                plt.scatter(
                    lon
                    ,lat
                    ,s=1
                    ,c=wv[idx]
                    ,transform=transf
                    ,vmin=vmin
                    ,vmax=vmax
                )
        plt.show()
