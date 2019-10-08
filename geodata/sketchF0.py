
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

workFileName = "sketchF.h5"
#workFile     = h5.File(workPath+workFileName,'r')
workFile     = h5.File(workFileName,'r')

# for i in range(10):
#     print(i,hex(workFile['/image']['stare_temporal'][i]))

# si0 = 1000000
# si1 = 1000100
# ss = workFile['/image']['stare_spatial'][si0:si1]
# st = workFile['/image']['stare_temporal'][si0:si1]
# sc = workFile['/image']['goes_src_coord'][si0:si1]
# print(si0,'si,s,t,c: ',ss,st,sc)

nx = workFile['/image_description']['nx']
ny = workFile['/image_description']['ny']
print('(nx,ny): ',(nx,ny))

fig, (ax0,ax1,ax2,ax3) = plt.subplots(nrows=4)

b3_img = workFile['/image']['goes_b3'].reshape(ny,nx)
print('b3_img shape: ',b3_img.shape)
print('b3 mnmx: ',np.amin(b3_img),np.amax(b3_img))
ax0.set_title('b3')
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax0.imshow(b3_img)

b4_img = workFile['/image']['goes_b4'].reshape(ny,nx)
print('b4 mnmx: ',np.amin(b4_img),np.amax(b4_img))
ax1.set_title('b4')
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.imshow(b4_img)

b5_img = workFile['/image']['goes_b5'].reshape(ny,nx)
print('b5 mnmx: ',np.amin(b5_img),np.amax(b5_img))
ax2.set_title('b5')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.imshow(b5_img)

tpw_scale  = workFile['/merra2_description']['tpw_scale']
tpw_offset = workFile['/merra2_description']['tpw_offset']
print('tpw scale offset: ',tpw_scale,tpw_offset)

# m2_img = workFile['/image']['merra2_tpw'].reshape(nx,ny)
m2_img = tpw_offset + tpw_scale*workFile['/image']['merra2_tpw'].reshape(ny,nx)
print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))
ax3.set_title('tpw')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax3.imshow(m2_img)
plt.show()

fig,ax = plt.subplots(1,1)
# print('ax ',ax,type(ax))
print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))
ax.set_title('tpw')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.imshow(m2_img)
plt.show()

workFile.close()
