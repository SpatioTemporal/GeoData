

"""
Load a GOES 10 band 5 image.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from netCDF4 import Dataset

dataPath = "/home/mrilee/data/"
# dataFile = "goes10.2005.349.000122.sndr.BAND_07.nc" 
dataFile = "goes10.2005.349.003015.BAND_05.nc"

fqFilename = dataPath+dataFile

ds = Dataset(fqFilename)

# lat  = ds['lat'][:,:]
# lon  = ds['lon'][:,:]
# data = ds['data'][0,:,:]

lat0=0
lat1=1355
lon0=0
lon1=3311

# lat0=500
# lat1=1355
# lon0=0
# lon1=3311

lat  = ds['lat'][lat0:lat1,lon0:lon1]
lon  = ds['lon'][lat0:lat1,lon0:lon1]
data = ds['data'][0,lat0:lat1,lon0:lon1]

print('lat.shape:  ',lat.shape)
print('lon.shape:  ',lon.shape)
print('data.shape: ',data.shape)


plt.figure()
plt.imshow(data)
plt.show()

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.coastlines()
# plt.contourf(lon,lat,data,60,transform=ccrs.PlateCarree())
plt.scatter(lon,lat,s=20,c=data)

plt.show()
