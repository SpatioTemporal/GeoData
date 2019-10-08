
# Read h5 file and try CCL.

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

import cv2
from ccl_marker_stack import ccl_marker_stack

# workFileName = "work.h5"
workFileName = "sketchF.h5"
#workFile     = h5.File(workPath+workFileName,'r')
workFile     = h5.File(workFileName,'r')

tpw_scale  = workFile['/merra2_description']['tpw_scale']
tpw_offset = workFile['/merra2_description']['tpw_offset']
print('tpw scale offset: ',tpw_scale,tpw_offset)

b5_img = workFile['/image']['goes_b5']
# b5_img = workFile['/image']['goes_b4']
# b5_img = workFile['/image']['goes_b3']
print('b5 mnmx: ',np.amin(b5_img),np.amax(b5_img))

# m2_img = workFile['/image']['merra2_tpw']
m2_img = tpw_offset + tpw_scale*workFile['/image']['merra2_tpw']
print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))

# b5_lo,b5_threshold = (7500.0,15000.0)
# b5_lo,b5_threshold = (7500.0,12500.0)
b5_lo,b5_threshold = (0.0,8000.0) # b5
# b5_lo,b5_threshold = (0.0,5000.0) # b3
# b5_lo,b5_threshold = (1000.0,8000.0)

b5_img_ge2_idx = np.where((b5_img <= b5_threshold) & (b5_img>b5_lo)) # This is where TPW is high and b5 is low.

b5_img_lt2_idx = np.where((b5_img >  b5_threshold) | (b5_img < b5_lo )) # Reverse.

nx = workFile['/image_description']['nx']
ny = workFile['/image_description']['ny']

b5_thresh=[b5_lo,b5_threshold]

# Copy the following from ccl2d.

if False:
    cv2.imshow('b5_img',np.array(255*b5_img.reshape([nx,ny])/np.amax(b5_img),dtype=np.uint8)); cv2.waitKey(0); cv2.destroyAllWindows()

mx   = np.nanmax(b5_img)
if mx == 0:
    mx = 1.0
data = np.array(255.0*b5_img/mx,dtype=np.uint8).reshape([nx,ny])
d_trigger      = int(255.0*b5_threshold/mx)
d_out          = int(255)

# Why does the limb show up in the thresh, but not the labels?

# Eliminate the sky.
data[np.where(data < (255*3000/mx))] = 255

print('d type: ',type(data))
print('d trigger,out: ',d_trigger,d_out)
print('d mnmx:  ',np.amin(data),np.amax(data))

if False:
    cv2.imshow('data',data); cv2.waitKey(0); cv2.destroyAllWindows()

# This works
if True:
    # Pass in external threshold
    marker_stack = ccl_marker_stack() 
    m0_new,m1_new,m0_eol,translation01\
        = marker_stack.make_slice_from(
            data
            ,(d_trigger,d_out)
            ,graph=False
            ,thresh_inverse=True
            ,global_latlon_grid=False
            ,norm_data=False
            ,perform_threshold=True)
    markers=m1_new
    print('markers type,len ',type(markers),len(markers))
    # print('markers ',markers)

thresh = None

# The following two also work
if False:
    ret,thresh  = cv2.threshold(data,d_trigger,d_out,cv2.THRESH_BINARY_INV) # less than, for b5
    print('thresh ret:  ',type(ret),ret)
    print('thresh type: ',type(thresh),thresh.shape,np.amin(thresh),np.amax(thresh))
    # Pass in data, ask for threshold
    marker_stack = ccl_marker_stack() 
    m0_new,m1_new,m0_eol,translation01\
        = marker_stack.make_slice_from(
            thresh
            ,(d_trigger,d_out)
            ,graph=False
            ,thresh_inverse=True
            ,global_latlon_grid=False
            ,norm_data=False
            ,perform_threshold=False)
    markers=m1_new
    print('markers type,len ',type(markers),len(markers))
    # print('markers ',markers)

if False:
    cv2.imshow('thresh',thresh); cv2.waitKey(0); cv2.destroyAllWindows()

if False:
    ret,markers = cv2.connectedComponents(thresh)

markers_mx = np.amax(markers)
print('markers_mx: ',markers_mx)

if markers_mx == 0:
    markers_mx = 1

data1=markers.astype(np.float)/markers_mx
print('markers',type(markers),type(data1))

if False:
    cv2.imshow('markers',data1); cv2.waitKey(0); cv2.destroyAllWindows()

# https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
# For fun viz.
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    # cv2.imshow('labeled.png', labeled_img); cv2.waitKey()
    return labeled_img

nrows = 5
fig,axs = plt.subplots(nrows=nrows)

for row in range(nrows):
    axs[row].get_xaxis().set_visible(False)
    axs[row].get_yaxis().set_visible(False)

axs[0].imshow(b5_img.reshape(nx,ny))
axs[1].imshow(data)
if thresh is not None:
    axs[2].imshow(thresh)
axs[3].imshow(markers)
axs[4].imshow(imshow_components(markers))
plt.show()




