import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from netCDF4 import Dataset

dataPath = "/home/mrilee/data/"
dataFile = "MERRA2_300.tavg1_2d_slv_Nx.20051215.nc4"

fqFilename = dataPath+dataFile

ds = Dataset(fqFilename)
print('keys: ',ds.variables.keys())

# lat  = ds['lat'][:,:]
# lon  = ds['lon'][:,:]
# data = ds['data'][0,:,:]

# MERRA-2
lat0=0
lat1=361+1
lon0=0
lon1=576+1
tim0=0
tim1=23+1

# lat0=500
# lat1=1355
# lon0=0
# lon1=3311

# lat     = ds['lat'][:]
lat     = ds['lat'][lat0:lat1]
# lon     = ds['lon'][:]
lon     = ds['lon'][lon0:lon1]
tim     = ds['time'][tim0:tim1]
dataDay = ds['TQV'][tim0:tim1,lat0:lat1,lon0:lon1]
data    = dataDay[10,:,:].T

latsg,lonsg = np.meshgrid(lat,lon)

print('lat.shape:      ',lat.shape)
print('lon.shape:      ',lon.shape)
print('tim.shape:      ',tim.shape)
print('latsg.shape:     ',latsg.shape)
print('lonsg.shape:     ',lonsg.shape)
print('dataDay.shape:  ',dataDay.shape)
print('data.shape:     ',data.shape)

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.coastlines()
# plt.contourf(lon,lat,data,60,transform=ccrs.PlateCarree())
# plt.scatter(lon,lat,s=20,c=data)
plt.scatter(lonsg,latsg,s=15,c=data)

plt.show()
