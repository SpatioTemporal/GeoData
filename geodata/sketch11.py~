

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

import pystare as ps

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
elemRes = ds['elemRes'][0]*0.5

def resolution(km):
    return 10-np.log2(km/10)
if(False):
    for i in range(28):
        l = 0.5*np.pi*6370.0/np.power(2,i)
        print(i,l,resolution(l))


print('lat.shape:    ',lat.shape)
print('lon.shape:    ',lon.shape)
print('data.shape:   ',data.shape)
print('elemRes (km): ',elemRes)
print('res(elemRes): ',resolution(elemRes))
print('type(lat[0]): ',type(lat[0,0]))

i0 = 500*lat1
n = 10
lat_flat = lat.flatten()[i0:i0+n]
# print('type(lat_flat[0]): ',type(lat_flat[0]))
lon_flat = lon.flatten()[i0:i0+n]
data_flat = data.flatten()[i0:i0+n]
indices = np.zeros([n],dtype=np.int64)
indices = ps.from_latlon(lat_flat,lon_flat,int(resolution(elemRes)))
indices = ps.from_latlon(lat_flat,lon_flat,int(resolution(elemRes)))
for i in range(n):
    print(i,lat_flat[i],lon_flat[i],hex(indices[i]))

# exit()

def shiftarg_lon(lon):
    "If lon is outside +/-180, then correct back."
    if(lon>180):
        return ((lon + 180.0) % 360.0)-180.0
    else:
        return lon

def triangulate1(lats,lons):
    "Prepare data for tri.Triangulate."
    print('triangulating1...')
    intmat=[]
    npts=int(len(lats)/3)
    k=0
    for i in range(npts):
        intmat.append([k,k+1,k+2])
        k=k+3
    for i in range(len(lons)):
        lons[i] = shiftarg_lon(lons[i])
    print('triangulating1 done.')      
    return lons,lats,intmat

def plot_indices(indices,c='r',transform=None,lw=1):
    latv,lonv,latc,lonc = ps.to_vertices_latlon(indices)
    lons,lats,intmat = triangulate1(latv,lonv)
    triang = tri.Triangulation(lons,lats,intmat)
    plt.triplot(triang,c+'-',transform=transform,lw=lw,markersize=3)

def bbox_lonlat(lat,lon,km,close=False):
    re = 6371.0
    delta = (km/6371.0)*180.0/np.pi
    latm=lat-delta
    latp=lat+delta
    lonm=lon-delta
    lonp=lon+delta
    if close:
        return [lonm,lonp,lonp,lonm,lonm],[latm,latm,latp,latp,latm]
    else:
        return [lonm,lonp,lonp,lonm],[latm,latm,latp,latp]

# def make_hull(lat0,lon0,resolution0,ntri0):
#     hull0 = ps.to_hull_range_from_latlon(lat0,lon0,resolution0,ntri0)
#     lath0,lonh0,lathc0,lonhc0 = ps.to_vertices_latlon(hull0)
#     lons0,lats0,intmat0 = triangulate1(lath0,lonh0)
#     triang0 = tri.Triangulation(lons0,lats0,intmat0)
#     return lats0,lons0,triang0,hull0

if(False):
    plt.figure()
    plt.imshow(data)
    plt.show()
if(False):
    plt.figure()
    plt.imshow(lat)
    plt.show()

# exit()

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.coastlines()
# plt.contourf(lon,lat,data,60,transform=ccrs.PlateCarree())
# plt.scatter(lon,lat,s=20,c=data)

plt.scatter(lon_flat,lat_flat,s=30,c=data_flat,transform=ccrs.Geodetic())
plot_indices(indices,transform=ccrs.Geodetic())

if True:
    print('Add circles around observation points.')
    for i in range(len(lon_flat)):
        lonplot,latplot = bbox_lonlat(lat_flat[i],lon_flat[i],elemRes,close=True)
        ax.plot(lonplot,latplot,True,transform=ccrs.Geodetic())
        ax.add_patch(plt.Circle((lon_flat[i],lat_flat[i]),radius=elemRes*180.0/(np.pi*6371.0),fill=False,transform=ccrs.Geodetic()))

if True:
    i = 2;
    print('Add a "bounding box" around point ',i)
    lonplot,latplot = bbox_lonlat(lat_flat[i],lon_flat[i],elemRes,close=False)
    hull = ps.to_hull_range_from_latlon(latplot,lonplot,13,300)
    # hull = ps.to_hull_range_from_latlon(latplot,lonplot,15,300) # Looks cool!
    plot_indices(hull,c='b',transform=ccrs.Geodetic(),lw=1.5)
    print('Compare the indices with the "hulled bounding box."')
    print('cmp len indices: ',len(indices),indices.shape)
    print('cmp len hull:    ',len(hull),hull.shape)
    cmp = ps.cmp_spatial(indices,hull)
    cmpr = ps.cmp_spatial(hull,indices)
    nindices = len(indices)
    nhull = len(hull)
    for i in range(nindices):
        for j in range(nhull):
            print(i,j,' i,j,cmp: ',(cmp[i*nhull+j],cmpr[j*nindices+i]),hex(indices[i]),hex(hull[j]))
    print('cmp Plot the intersecting triangles in a thicker red line.')
    plot_indices(hull[[18,19,20,21]],c='r',transform=ccrs.Geodetic(),lw=2)

# ax.add_patch(plt.Circle((lon_flat[i],lat_flat[i]),radius=elemRes*180.0/(np.pi*6371.0),color=None,fc=None,ec='c',fill=False))

plt.show()
