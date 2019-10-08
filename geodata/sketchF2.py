
import h5py as h5
from ccl_marker_stack import ccl_marker_stack, imshow_components

import numpy as np
import cv2
from ccl_marker_stack import ccl_marker_stack

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

data_dir = ''
data_filenames = [
    'sketchF1.0x007d5c684080008a.h5'
    ,'sketchF1.0x007d5c685e80008a.h5'
    ,'sketchF1.0x007d5c688080008a.h5'
    ,'sketchF1.0x007d5c689e80008a.h5'
]

marker_stack = ccl_marker_stack() 

# iDataFile = 0
for iDataFile in range(len(data_filenames)):

    print('adding ',data_filenames[iDataFile])
    
    workFile = h5.File(data_dir+data_filenames[iDataFile],'r')
    
    m2_tpw_scale  = workFile['/merra2_description']['tpw_scale']
    m2_tpw_offset = workFile['/merra2_description']['tpw_offset']
    m2_img = m2_tpw_offset + m2_tpw_scale*workFile['/image']['merra2_tpw']

    nx = workFile['/image_description']['nx']
    ny = workFile['/image_description']['ny']
    g_img       = workFile['/image']['goes_b5']

    print('nx,ny: ',nx,ny)

    g_img = g_img.reshape([ny,nx])
    nx=500; ny=500
    x0= 700; y0=200
    tmp  = np.zeros([ny,nx],dtype=np.int64)
    tmp[:,:]  = g_img[y0:y0+ny,x0:x0+nx]
    g_img = tmp.reshape([nx*ny])
    # g_img = tmp.flatten()

    g_lo        = 0.0
    g_threshold = 8000.0
    
    # g_img_ge2_idx = np.where((g_img <= g_threshold) & (g_img > g_lo)) # This is where TPW is high and b5 is low.
    # g_img_lt2_idx = np.where((g_img >  g_threshold) | (g_img < g_lo )) # Reverse.

    g_thresh=[g_lo,g_threshold]
    
    # Copy the following from ccl2d.
    
    mx        = np.nanmax(g_img)
    if mx == 0:
        mx = 1.0
    data      = np.array(255.0*g_img/mx,dtype=np.uint8).reshape([ny,nx])
    d_trigger = int(255.0*g_threshold/mx)
    d_out     = int(255)
    
    # Why does the limb show up in the thresh, but not the labels?
    
    # Eliminate the sky.
    data[np.where(data < (255*3000/mx))] = 255
    
    # The following two also work
    if True:
        ret,thresh  = cv2.threshold(data,d_trigger,d_out,cv2.THRESH_BINARY_INV) # less than, for g
        print('thresh ret:  ',type(ret),ret)
        print('thresh type: ',type(thresh),thresh.shape,np.amin(thresh),np.amax(thresh))
    
    if True:
        # Pass in data, ask for threshold
        # marker_stack = ccl_marker_stack() 
        print('adding slice...')
        m0_new,m1_new,m0_eol,translation01\
            = marker_stack.make_slice_from(
                data
                ,(d_trigger,d_out)
                ,graph=False
                ,thresh_inverse=True
                ,global_latlon_grid=False
                ,norm_data=False
                ,perform_threshold=True)
        print('done adding slice.')
        markers=m1_new
        print('markers type,len ',type(markers),len(markers))
        # print('markers ',markers)
    
    workFile.close()

    if False:
        nrows = 5
        fig,axs = plt.subplots(nrows=nrows)
        for row in range(nrows):
            axs[row].get_xaxis().set_visible(False)
            axs[row].get_yaxis().set_visible(False)
        axs[0].imshow(g_img.reshape(ny,nx))
        axs[1].imshow(data)
        axs[2].imshow(thresh)
        axs[3].imshow(markers)
        axs[4].imshow(imshow_components(markers))
        plt.show()

m_results_translated = marker_stack.resolve_labels_across_stack()
print('m_results_translated len:  ',len(m_results_translated))
print('m_results_translated typ:  ',type(m_results_translated))
print('m_results_translated typ1: ',type(m_results_translated[0]))
print('m_results_translated typ2: ',type(m_results_translated[0][0,0]))

nrows = len(m_results_translated)+1
fig,axs = plt.subplots(nrows=nrows)
for row in range(nrows):
    axs[row].get_xaxis().set_visible(False)
    axs[row].get_yaxis().set_visible(False)
for row in range(nrows-1):
    axs[row].imshow(m_results_translated[row])
axs[-1].imshow(g_img.reshape([ny,nx]))
print('plotting final')
plt.show()

    
    
