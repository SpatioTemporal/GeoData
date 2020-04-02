#!/usr/bin/env python

###########################################################################
# Read a vds made by sketchJ1.py and display.
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
    fname = 'MOD05_L2.A2005349.2120.061.2017294065852.wv_nir.sketchJ0.vds.h5'
    with h5.File(fname,'r') as h:
        shape = (h['metadata']['shape1'][0],h['metadata']['shape0'][0],)
        img   = h['wv_nir']['Water_Vapor_Near_Infrared']
        print('img ',type(img),img.dtype,img.shape,shape)
        img   = img.reshape(shape)
        plt.imshow(img,origin='lower',aspect=1)
        plt.show()
    return

if __name__ == "__main__":
    main()


## from matplotlib import pyplot as plt
## plt.imshow(data, interpolation='nearest')
## plt.show()
