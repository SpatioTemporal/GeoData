import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from netCDF4 import Dataset

dataPath = "/home/mrilee/data/"

# dataFile = "NPR.AAOP.NK.D05348.S2222.E0015.B3944850.NS" # TPW
# dataKey='TPW'

# dataFile = "NPR.DMOP.S7.D05348.S2343.E0126.B0055363.NS" # TPW
# dataKey='TPW'

dataFile = "NPR.MHOP.NN.D05348.S2332.E0113.B0294041.NS" # No TPW
dataKey='Chan5_AT'

fqFilename = dataPath+dataFile

ds = Dataset(fqFilename)
print('keys: ',ds.variables.keys())

shape = ds['Latitude'].shape

# MERRA-2
y0=0
y1=shape[1]
x0=0
x1=shape[0]
tim0=0
tim1=shape[0]

print('x01: ',x0,x1)
print('y01: ',y0,y1)
print('t01: ',tim0,tim1)

print('lat shape: ',ds['Latitude'].shape)

y     = ds['Latitude'][:,:]
x     = ds['Longitude'][:,:]
# tim     = ds['Time'][tim0:tim1:]
# dataDay = ds[dataKey][y0:y1,x0:x1]
dataDay = ds[dataKey][:,:]
data    = dataDay[:,:]

# xg,yg = np.meshgrid(x,y)
xg,yg = x,y

print('x.shape:        ',x.shape)
print('y.shape:        ',y.shape)
# print('tim.shape:      ',tim.shape)
print('xg.shape:     ',xg.shape)
print('yg.shape:     ',yg.shape)
print('dataDay.shape:  ',dataDay.shape)
print('data.shape:     ',data.shape)

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.coastlines()
# plt.contourf(x,y,data,60,transform=ccrs.PlateCarree())
# plt.scatter(x,y,s=20,c=data)
plt.scatter(x,y,s=1.25,c=data)
# plt.scatter(x,y,s=0.5,c=data)

plt.show()
