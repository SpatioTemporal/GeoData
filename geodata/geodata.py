
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

import numpy as np
import pystare as ps
import unittest


re_km = 6371.0 # radius earth km
deg_per_rad = 180.0/np.pi

precipitable_water_cm_per_kgom2 = 0.1/1.0

def shiftarg_lon(lon):
    "If lon is outside +/-180, then correct back."
    if(lon>180):
        return ((lon + 180.0) % 360.0)-180.0
    else:
        return lon

def resolution(km):
    return 10-np.log2(km/10)

def triangulate1(lats,lons):
    "Prepare data for tri.Triangulate."
    # print('triangulating1...')
    intmat=[]
    npts=int(len(lats)/3)
    k=0
    for i in range(npts):
        intmat.append([k,k+1,k+2])
        k=k+3
    for i in range(len(lons)):
        lons[i] = shiftarg_lon(lons[i])
    # print('triangulating1 done.')      
    return lons,lats,intmat

def plot_indices(indices,c='r',transform=None,lw=1):
    latv,lonv,latc,lonc = ps.to_vertices_latlon(indices)
    lons,lats,intmat = triangulate1(latv,lonv)
    triang = tri.Triangulation(lons,lats,intmat)
    plt.triplot(triang,c+'-',transform=transform,lw=lw,markersize=3)
    return

def bbox_lonlat(lat,lon,km,close=False):
  if type(km) in map(type,[[],()]):
    delta_lat = (km[0]/re_km)*deg_per_rad
    delta_lon = (km[1]/re_km)*deg_per_rad
  else:
    delta_lat = (km/re_km)*deg_per_rad
    delta_lon = (km/re_km)*deg_per_rad
  latm=lat-delta_lat
  latp=lat+delta_lat
  lonm=lon-delta_lon
  lonp=lon+delta_lon
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

if __name__ == '__main__':

  print('running')

# if True:
#   unittest.main()
        
