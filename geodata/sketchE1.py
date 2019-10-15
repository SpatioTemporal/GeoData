
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

workFileNames = [
    'sketchF1.0x007d5c684080008a.h5'
    ,'sketchF1.0x007d5c685e80008a.h5'
    ,'sketchF1.0x007d5c688080008a.h5'
    ,'sketchF1.0x007d5c689e80008a.h5' ]

workFileName = workFileNames[0]
# workFileName = "work.h5"
# workFileName = "sketchF.h5"
# workFile     = h5.File(workPath+workFileName,'r')
workFile     = h5.File(workFileName,'r')

tpw_scale  = workFile['/merra2_description']['tpw_scale']
tpw_offset = workFile['/merra2_description']['tpw_offset']
# print('tpw scale offset: ',tpw_scale,tpw_offset)

# b3_img = workFile['/image']['goes_b3']
b4_img = workFile['/image']['goes_b4']
b5_img = workFile['/image']['goes_b5']

# print('b5 mnmx: ',np.amin(b5_img),np.amax(b5_img))

## # m2_img = workFile['/image']['merra2_tpw']
m2_img = tpw_offset + tpw_scale*workFile['/image']['merra2_tpw']
## print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))

##  # b5_lo,b5_threshold = (7500.0,15000.0)
##  # b5_lo,b5_threshold = (7500.0,12500.0)
##  b5_lo,b5_threshold = (0.0,8000.0) # b5
##  # b5_lo,b5_threshold = (0.0,5000.0) # b3
##  # b5_lo,b5_threshold = (1000.0,8000.0)
##  b5_img_ge2_idx = np.where((b5_img <= b5_threshold) & (b5_img>b5_lo)) # This is where TPW is high and b5 is low.
##  b5_img_lt2_idx = np.where((b5_img >  b5_threshold) | (b5_img < b5_lo )) # Reverse.

nx = workFile['/image_description']['nx']
ny = workFile['/image_description']['ny']

bx_lo = 1000
# b45_threshold = -1850 # for m2_threshold = ~20
# b45_thresh=[b45_lo,b45_threshold]

b45_tpw_scale  =    17.6
b45_tpw_offset = -2196
m2_threshold   =    50.0
b45_threshold = b45_tpw_scale*m2_threshold + b45_tpw_offset

# Mask to valid and shift data values to zero
idx_valid   = np.where((b5_img>bx_lo) & (b4_img>bx_lo))
idx_invalid = np.where((b5_img<=bx_lo) & (b4_img<=bx_lo))
b45_img = np.full(b5_img.shape,0,dtype=b5_img.dtype)
# Shift to zero
b45_img[idx_valid] = b4_img[idx_valid]-b5_img[idx_valid]-b45_tpw_offset # Shift to zero
b45_threshold = b45_threshold - b45_tpw_offset # Shift to zero
# Slam the negative
b45_img[np.where(b45_img<0)] = 0 # Slam the negative. Sorry. Mea culpa.

# Print summary to check
print('*** summary ***')
print('m2_thresdhold = %f kgom2, b45_threshold = %f d-counts-zeroed, %f d-counts '%(m2_threshold,b45_threshold,b45_threshold+b45_tpw_offset))
print('b45_img mnmx: ',np.amin(b45_img),np.amax(b45_img))

# Copy the following from ccl2d.

if False:
    b45_viz = np.full(b45_img.shape,np.amin(b45_img[idx_valid])-1,dtype=b45_img.dtype)
    b45_viz[idx_valid]=b45_img[idx_valid]
    cv2.imshow('b45_img',np.array(255*b45_viz.reshape([ny,nx])/np.amax(b45_img),dtype=np.uint8)); cv2.waitKey(0); cv2.destroyAllWindows()

if False:
    b45_img       = m2_img
    b45_threshold = 30.0

def mnmx_masked(x,idx):
    return (np.min(x[idx]),np.max(x[idx]))

def scale_to_uint8(x,mnmx=None,offset=0):
    if mnmx is None:
        mn = np.nanmin(x)
        mx = np.nanmax(x)
        mnmx = (mn,mx)
    else:
        mn=mnmx[0]
        mx=mnmx[1]
    if mx-mn == 0:
        mx = mn + 1
    xs = offset+((255.0-offset)*(x-mn))/(mx-mn)
    return np.uint8(xs),mnmx

if idx_valid is not None:
    data_scaled,b45_img_mnmx = scale_to_uint8(b45_img,mnmx_masked(b45_img,idx_valid)); 
else:
    data_scaled,b45_img_mnmx = scale_to_uint8(b45_img)

mnmx = b45_img_mnmx
data = np.zeros(data_scaled.shape,dtype=np.uint8)
data = data_scaled.reshape(ny,nx)
# data = data_scaled
# data[np.where(b5_img > 1000)] = data_scaled[np.where(b5_img > 1000)]
# data = data.reshape(ny,nx)
d_trigger      = int(255.0*(b45_threshold-mnmx[0])/(mnmx[1]-mnmx[0]))
d_out          = int(255)

# Why does the limb show up in the thresh, but not the labels?

# Eliminate the sky.
# data[np.where(data < (255*3000/mx))] = 255

print('d type: ',type(data))
print('d trigger,out: ',d_trigger,d_out)
print('d mnmx:  ',np.amin(data),np.amax(data))

if False:
    cv2.imshow('data',data); cv2.waitKey(0); cv2.destroyAllWindows()

# This works
if True:
    # Pass in data, ask for threshold
    marker_stack = ccl_marker_stack(global_latlon_grid=False) 
    m0_new,m1_new,m0_eol,translation01\
        = marker_stack.make_slice_from(
            data
            ,(d_trigger,d_out)
            ,graph=False
            ,thresh_inverse=False
            ,norm_data=False
            ,perform_threshold=True)
    markers=m1_new
    print('markers type,len ',type(markers),len(markers))
    # print('markers ',markers)

thresh = None

# The following two also work
if False:
    # Pass in external threshold
    # ret,thresh  = cv2.threshold(data,d_trigger,d_out,cv2.THRESH_BINARY_INV) # less than, for b5
    ret,thresh  = cv2.threshold(data,d_trigger,d_out,cv2.THRESH_BINARY) # less than, for b5
    print('thresh ret:  ',type(ret),ret)
    print('thresh type: ',type(thresh),thresh.shape,np.amin(thresh),np.amax(thresh))
    marker_stack = ccl_marker_stack(global_latlon_grid=False) 
    m0_new,m1_new,m0_eol,translation01\
        = marker_stack.make_slice_from(
            thresh
            ,(d_trigger,d_out)
            ,graph=False
            ,thresh_inverse=False
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

nrows = 3
ncols = 2
fig,axs = plt.subplots(nrows=nrows,ncols=ncols)

for irow in range(nrows):
    for icol in range(ncols):
        axs[irow,icol].get_xaxis().set_visible(False)
        axs[irow,icol].get_yaxis().set_visible(False)

irow,icol = (0,0)
axs[irow,icol].imshow(b45_img.reshape(ny,nx))
irow,icol = (1,0)
axs[irow,icol].imshow(data)
if thresh is not None:
    irow,icol = (2,0)
    axs[irow,icol].imshow(thresh)
irow,icol = (0,1)
axs[irow,icol].imshow(m2_img.reshape(ny,nx))
irow,icol = (1,1)
axs[irow,icol].imshow(markers)
irow,icol = (2,1)
axs[irow,icol].imshow(imshow_components(markers))

plt.show()


nrows = 2
ncols = 2
fig,axs = plt.subplots(nrows=nrows,ncols=ncols)
for irow in range(nrows):
    for icol in range(ncols):
        axs[irow,icol].get_xaxis().set_visible(False)
        axs[irow,icol].get_yaxis().set_visible(False)

irow,icol = (0,0)
axs[irow,icol].imshow(m2_img.reshape(ny,nx))

irow,icol = (1,0)
axs[irow,icol].imshow(imshow_components(markers))

irow,icol = (1,1)
axs[irow,icol].imshow(b45_img.reshape(ny,nx))

plt.show()

