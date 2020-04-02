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



