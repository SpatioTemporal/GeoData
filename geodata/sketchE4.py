
###########################################################################

import geodata as gd
import h5py as h5
from netCDF4 import Dataset
import numpy as np
import pystare as ps

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

import yaml

from stopwatch import sw_timer

import cv2
from ccl_marker_stack import ccl_marker_stack

###########################################################################
# https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
# For fun viz.
def imshow_components(labels):
    # Map component labels to hue val
    # label_hue = np.uint8(179*labels/np.max(labels))
    label_hue = np.uint8(179*labels/64)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    # cv2.imshow('labeled.png', labeled_img); cv2.waitKey()
    return labeled_img

###########################################################################

final_markers_filenames=\
['sketchE3.sketchF1.0x007d5c684080008a-final-markers-1.h5'
,'sketchE3.sketchF1.0x007d5c685e80008a-final-markers-1.h5'
,'sketchE3.sketchF1.0x007d5c688080008a-final-markers-1.h5'
,'sketchE3.sketchF1.0x007d5c689e80008a-final-markers-1.h5']

final_markers_intermediate_filenames=\
['sketchE3.sketchF1.0x007d5c684080008a-markers-1.h5'
,'sketchE3.sketchF1.0x007d5c685e80008a-markers-1.h5'
,'sketchE3.sketchF1.0x007d5c688080008a-markers-1.h5'
,'sketchE3.sketchF1.0x007d5c689e80008a-markers-1.h5']

nrows=len(final_markers_filenames)
fig,axs = plt.subplots(nrows=nrows)
for iter in range(nrows):
    print('visualizing iter = ',iter)
    workFile = final_markers_filenames[iter]
    inFile  = h5.File(workFile,'r')
    markers = inFile['/markers']['markers'].reshape(inFile['/markers_description']['ny'],inFile['/markers_description']['nx'])
    inFile.close()
    ax=axs[iter]
    ax.set_title(workFile)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(imshow_components(markers))
plt.show()

