
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

workDir = "/home/mrilee/opt/src/mrilee/git/Fumarole/fumarole/"
# workFileName = workDir+"sketch0.0x007d5c681ebc008a.h5" # ok
workFileName = workDir+"sketch0.0x007d5c6840bc008a.h5" # bad merra2?
# workFileName = workDir+"sketch0.0x007d5c685ebc008a.h5" # ok
# workFileName = workDir+"sketch0.0x007d5c6880bc008a.h5" # bad
# workFileName = workDir+"sketch0.0x007d5c689ebc008a.h5" # ok

# LOCAL FILES
# workFileName = "work.h5"
# workFileName = "sketch9.2005.349.213015.h5"
# workFileName = "sketchF.h5"
# workFileName = "sketchF1.0x007d5c684080008a.h5"
# workFileName = "sketchF1.0x007d5c685e80008a.h5"
# workFileName = "sketchF1.0x007d5c688080008a.h5"
# workFileName = "sketchF1.0x007d5c689e80008a.h5"
print('sketch90 loading ',workFileName)
workFile     = h5.File(workFileName,'r')

#workFile     = h5.File(workPath+workFileName,'r')

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

fig, (ax0,ax1) = plt.subplots(nrows=2)

b5_img = workFile['/image']['goes_b5'].reshape(ny,nx)
print('b5 mnmx: ',np.amin(b5_img),np.amax(b5_img))
ax0.set_title('b5')
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax0.imshow(b5_img)

tpw_scale  = workFile['/merra2_description']['tpw_scale']
tpw_offset = workFile['/merra2_description']['tpw_offset']
print('tpw scale offset: ',tpw_scale,tpw_offset)

# m2_img = workFile['/image']['merra2_tpw'].reshape(ny,nx)
m2_img = tpw_offset + tpw_scale*workFile['/image']['merra2_tpw'].reshape(ny,nx)
print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))
ax1.set_title('tpw')
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.imshow(m2_img)
plt.show()

fig,ax = plt.subplots(1,1)
# print('ax ',ax,type(ax))
print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))
ax.set_title('tpw')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.imshow(m2_img)
plt.show()


fig, axs = plt.subplots(nrows=4)
iax=0
axs[iax].set_title('tpw')
axs[iax].get_xaxis().set_visible(False)
axs[iax].get_yaxis().set_visible(False)
axs[iax].imshow(m2_img)

iax=1
b3_img = workFile['/image']['goes_b3'].reshape(ny,nx)
print('b3 mnmx: ',np.amin(b3_img),np.amax(b3_img))
axs[iax].set_title('b3')
axs[iax].get_xaxis().set_visible(False)
axs[iax].get_yaxis().set_visible(False)
axs[iax].imshow(b3_img)

iax=2
b4_img = workFile['/image']['goes_b4'].reshape(ny,nx)
print('b4 mnmx: ',np.amin(b4_img),np.amax(b4_img))
axs[iax].set_title('b4')
axs[iax].get_xaxis().set_visible(False)
axs[iax].get_yaxis().set_visible(False)
axs[iax].imshow(b4_img)

iax=3
#b5_img = workFile['/image']['goes_b5'].reshape(ny,nx)
print('b5 mnmx: ',np.amin(b5_img),np.amax(b5_img))
axs[iax].set_title('b5')
axs[iax].get_xaxis().set_visible(False)
axs[iax].get_yaxis().set_visible(False)
axs[iax].imshow(b5_img)

plt.show()

fig, axs = plt.subplots(nrows=2)

iax=0
# b3_img = workFile['/image']['goes_b3'].reshape(ny,nx)
# print('b3 mnmx: ',np.amin(b3_img),np.amax(b3_img))
axs[iax].set_title('b4-b5')
axs[iax].get_xaxis().set_visible(False)
axs[iax].get_yaxis().set_visible(False)
axs[iax].imshow(b4_img-b5_img)

iax=1
axs[iax].set_title('tpw')
axs[iax].get_xaxis().set_visible(False)
axs[iax].get_yaxis().set_visible(False)
axs[iax].imshow(m2_img)

plt.show()


workFile.close()
