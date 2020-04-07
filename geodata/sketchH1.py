#!/usr/bin/env python

# Check data written to vds-style file

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

###########################################################################
#
def main():
    return

if __name__ == '__main__':
    main()

    with h5.File("vds.h5",'r') as f:
        if False:
            ds = f['wv_nir']
            print('len ds dtype:  ',ds.dtype)
            print('len ds shape:  ',ds.shape)
        if True:
            md = f['metadata']
            print('len md dtype:  ',md.dtype)
            print('len md shape:  ',md.shape)
            print('md:            ',md)
            for i in range(md.shape[0]):
                print(i,' n_data=% 6s, src_name=%s in sare_id=0x%016x.'%(md['n_data'][i][0],md['src_name'][i][0],md['sare_id'][i][0]))

                
