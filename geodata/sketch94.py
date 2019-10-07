
# 

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

from scipy.stats import norm,skewnorm

workFileName = "work.h5"
#workFile     = h5.File(workPath+workFileName,'r')
workFile     = h5.File(workFileName,'r')

tpw_scale  = workFile['/merra2_description']['tpw_scale']
tpw_offset = workFile['/merra2_description']['tpw_offset']
print('tpw scale offset: ',tpw_scale,tpw_offset)

b5_img = workFile['/image']['goes_b5']
print('b5 mnmx: ',np.amin(b5_img),np.amax(b5_img))

# m2_img = workFile['/image']['merra2_tpw']
m2_img = tpw_offset + tpw_scale*workFile['/image']['merra2_tpw']
print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))

tpw_threshold = 20.0
# tpw_threshold = 25.0
# tpw_threshold = 30.0
# tpw_threshold = 35.0

b5_img_tot = b5_img[np.where(b5_img>1000)]
m2_img_ge2_idx = np.where((m2_img >= tpw_threshold) & (b5_img>1000)) # This is where TPW is high and b5 is low.
m2_img_lt2_idx = np.where((m2_img < tpw_threshold) & (b5_img>1000))  # Reverse.

nx = workFile['/image_description']['nx']
ny = workFile['/image_description']['ny']

### ### FIGURES ### 
fig,axs = plt.subplots(nrows=3,ncols=3)
# print('axs: ',axs)

for col in range(3):
    for row in range(3):
        axs[row,col].get_xaxis().set_visible(False)
        axs[row,col].get_yaxis().set_visible(False)

b5_img_ge = b5_img.copy()
b5_img_lt = b5_img.copy()
b5_img_ge[m2_img_lt2_idx]=0
b5_img_lt[m2_img_ge2_idx]=0

m2_img_ge = m2_img.copy()
m2_img_lt = m2_img.copy()
m2_img_ge[m2_img_lt2_idx]=0
m2_img_lt[m2_img_ge2_idx]=0

axs[0,0].set_title('goes b5')
axs[0,0].imshow(b5_img.reshape(nx,ny))

axs[1,0].set_title('goes b5(m2 tpw>%3.1f)'%tpw_threshold)
axs[1,0].imshow(b5_img_ge.reshape(nx,ny))

axs[2,0].set_title('goes b5(m2 tpw<%3.1f)'%tpw_threshold)
axs[2,0].imshow(b5_img_lt.reshape(nx,ny))

axs[0,1].set_title('m2 tpw')
axs[0,1].imshow(m2_img.reshape(nx,ny))

axs[1,1].set_title('m2 tpw>%3.1f'%tpw_threshold)
axs[1,1].imshow(m2_img_ge.reshape(nx,ny))

axs[2,1].set_title('m2 tpw<%3.1f'%tpw_threshold)
axs[2,1].imshow(m2_img_lt.reshape(nx,ny))

plt.show()
