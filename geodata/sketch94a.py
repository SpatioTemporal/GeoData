
# Read h5 file and display image masked by b5 thresholds.

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

# b5_threshold = 15000.0
# b5_threshold = 14000.0

# b5_lo,b5_threshold = (7500.0,12500.0)
# b5_lo,b5_threshold = (7500.0,10000.0)
# b5_lo,b5_threshold = (6000.0,10000.0)
# b5_lo,b5_threshold = (5000.0,10000.0)
# b5_lo,b5_threshold = (2500.0,10000.0)
# b5_lo,b5_threshold = (2500.0,12500.0)
# b5_lo,b5_threshold = (2500.0,7500.0)
# b5_lo,b5_threshold = (2500.0,8000.0)
# b5_lo,b5_threshold = (2500.0,8000.0)
b5_lo,b5_threshold = (0.0,8000.0)
# b5_lo,b5_threshold = (0.0,9000.0)

# b5_img_ge2_idx = np.where((b5_img <= b5_threshold) & (b5_img>1000)) # This is where TPW is high and b5 is low.

b5_img_ge2_idx = np.where((b5_img <= b5_threshold) & (b5_img>b5_lo)) # This is where TPW is high and b5 is low.

b5_img_lt2_idx = np.where((b5_img >  b5_threshold) | (b5_img < b5_lo )) # Reverse.

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
b5_img_ge[b5_img_lt2_idx]=0
b5_img_lt[b5_img_ge2_idx]=0

m2_img_ge = m2_img.copy()
m2_img_lt = m2_img.copy()
m2_img_ge[b5_img_lt2_idx]=-1e-3
m2_img_lt[b5_img_ge2_idx]=-1e-3

axs[0,0].set_title('goes b5')
axs[0,0].imshow(b5_img.reshape(nx,ny))

axs[1,0].set_title('goes %3.1fk<b5<%3.1fk'%(b5_lo/1000,b5_threshold/1000))
axs[1,0].imshow(b5_img_ge.reshape(nx,ny))

axs[2,0].set_title('goes b5>%3.1fk'%(b5_threshold/1000))
axs[2,0].imshow(b5_img_lt.reshape(nx,ny))

axs[0,1].set_title('m2 tpw')
axs[0,1].imshow(m2_img.reshape(nx,ny))

axs[1,1].set_title('m2 tpw %3.1fk<b5<%3.1fk'%(b5_lo/1000,b5_threshold/1000))
axs[1,1].imshow(m2_img_ge.reshape(nx,ny))

axs[2,1].set_title('m2 tpw b5>%3.1fk'%(b5_threshold/1000))
axs[2,1].imshow(m2_img_lt.reshape(nx,ny))

plt.show()
