
from math import modf

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

from netCDF4 import Dataset

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

stare_temporal_resolutions = { 2: {'1year':18,'1day':27, '1/2day':28,'4hr':30,'1hr':32,'1/2hr':33,'1/4hr':34,'1sec':44,'1msec':54}}

def format_time(yr,mo,dy,hr,mn,sc):
    return "%04d-%02d-%02dT%02d:%02d:%02d"%(yr,mo,dy,hr,mn,sc)

def merra2_parse_begin_date(tim):
    yr = int(tim/10000)
    mo = int(tim/100)-int(yr*100)
    dy = int(tim-mo*100)-int(yr*10000)
    return yr,mo,dy

def merra2_make_time(start_min,tim_min):
    start_hr = int(start_min / 60)
    start_mn = int(start_min % 60)
    tim_hr = int(tim_min / 60)
    tim_mn = int(tim_min % 60)
    t_hr = start_hr + tim_hr
    t_mn = start_mn + tim_mn
    return t_hr,t_mn

def merra2_stare_time(ds,iTime=None,tType=2,centered=True):
  if centered:
    start_time_mn = ds['time'].begin_time/100
    start_time_sec = ds['time'].begin_time % 100
    resolution = stare_temporal_resolutions[tType]['1/2hr']
  else:
    start_time_mn = 0;
    start_time_sec = 0
    resolution = stare_temporal_resolutions[tType]['1hr']
  yr,mo,dy = merra2_parse_begin_date(ds['time'].begin_date)
  tm = []
  if iTime is None:
    i0=0; i1=24
  else:
    i0=iTime; i1=iTime+1
  for i in range(i0,i1):
    # hr,mn    = merra2_make_time(ds['time'].begin_date,ds['time'][i])
    hr,mn    = merra2_make_time(start_time_mn,ds['time'][i])
    sc       = start_time_sec
    tm.append(format_time(yr,mo,dy,hr,mn,sc))
  dt       = np.array(tm,dtype='datetime64[ms]')
  idx      = ps.from_utc(dt.astype(np.int64),resolution)
  return idx

def merra2_stare_time_ds(ds):
  dt = merra2_stare_time(ds,iTime=12,centered=False);
  return stare_set_temporal_resolution(dt,stare_temporal_resolutions[2]['1/2day'])[0]

def goes10_img_stare_time(ds,tType=2,centered=True):
  resolution = stare_temporal_resolutions[2]['1/4hr']
  dt = np.array(ds['time'][0]*1000,dtype='datetime64[ms]').reshape([1])
  return ps.from_utc(dt.astype(np.int64),resolution)
  # return ps.from_utc(np.array(ds['time'][:]*1000,dtype='datetime64[ms]').astype(np.int64),resolution)

def datetime_from_stare(tId):
  if type(tId) is np.ndarray:
    return np.array(ps.to_utc_approximate(tId),dtype='datetime64[ms]')
  return np.array(ps.to_utc_approximate(np.array([tId],dtype=np.int64)),dtype='datetime64[ms]')[0]

def stare_set_temporal_resolution(tId,new_resolution):
  return (tId & ~(63*4))+(new_resolution*4)

def temporal_id_from_file(path,fname):
  ds = Dataset(path+fname)
  if "MERRA" in fname:
    return merra2_stare_time_ds(ds)
  elif "goes" in fname:
    return goes10_img_stare_time(ds)[0]
  else:
    return -1


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
        
