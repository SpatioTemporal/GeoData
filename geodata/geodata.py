

# Routines applying STARE to various Earth Science data sets.

# geodata.py

import datetime as dt

from math import modf

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

from netCDF4 import Dataset

import numpy as np
import pystare as ps
import os, fnmatch
import yaml
import unittest

try:
    from .modis_coarse_to_fine_geolocation import modis_5km_to_1km_geolocation as pascal_modis
except ImportError:
    from modis_coarse_to_fine_geolocation import modis_5km_to_1km_geolocation as pascal_modis

from collections import OrderedDict

###########################################################################
# A few constants
#
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

stare_temporal_resolutions = { 2: {'1year':18,'1day':27, '1/2day':28,'4hr':30,'1hr':32,'1/2hr':33,'1/4hr':34,'1/8hr':35,'1/16hr':36,'1sec':44,'1msec':54}}

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
    return None

def temporal_id_centered_from_goes_filename(gfname):
    gfname_split = gfname.split(".")
    yr  = gfname_split[1]
    ydy = gfname_split[2]
    hr  = int(gfname_split[3][0:2])
    mn  = int(gfname_split[3][2:4])
    sec = int(gfname_split[3][4:6])
    gdt = dt.datetime(int(yr),1,1)+dt.timedelta(int(ydy)-1)
    gdt_str = format_time(int(yr),int(gdt.month),int(gdt.day),hr,mn,sec)
    gdt_np = np.array([gdt_str],dtype='datetime64[ms]')
    gtid_centered = ps.from_utc(gdt_np.astype(np.int64),stare_temporal_resolutions[2]['1/4hr'])
    return gtid_centered

def temporal_id_centered_from_merra2_filename(m2name):
    m2name_split = m2name.split(".")
    yr  = int(m2name_split[2][0:4])
    mo  = int(m2name_split[2][4:6])
    dy  = int(m2name_split[2][6:8])
    hr  = 12
    mn  = 0
    sec = 0
    m2dt_str = format_time(yr,mo,dy,hr,mn,sec)
    m2dt_np = np.array([m2dt_str],dtype='datetime64[ms]')
    m2tid_centered = ps.from_utc(m2dt_np.astype(np.int64),stare_temporal_resolutions[2]['1/2day'])
    return m2tid_centered

def temporal_id_centered_from_modis_filename(mfname):
    mfname_split = mfname.split(".")
    yr  = mfname_split[1][1:5]
    ydy = mfname_split[1][5:8]
    hr  = int(mfname_split[2][0:2])
    mn  = int(mfname_split[2][2:4])
    # sec = int(mfname_split[2][4:6])
    sec = 0
    mdt = dt.datetime(int(yr),1,1)+dt.timedelta(int(ydy)-1)
    mdt_str = format_time(int(yr),int(mdt.month),int(mdt.day),hr,mn,sec)
    mdt_np = np.array([mdt_str],dtype='datetime64[ms]')
    mtid_centered = ps.from_utc(mdt_np.astype(np.int64),stare_temporal_resolutions[2]['1/16hr'])
    return mtid_centered

def temporal_id_centered_from_filename(fname):
  if "MERRA" in fname:
    return temporal_id_centered_from_merra2_filename(fname)[0]
  elif "goes" in fname:
    return temporal_id_centered_from_goes_filename(fname)[0]
  elif "MOD" in fname or "MYD" in fname:
    return temporal_id_centered_from_modis_filename(fname)[0]
  else:
    return None

def temporal_id_centered_filename_index(filenames):
    index = {}
    for entry in filenames:
        tid = temporal_id_centered_from_filename(entry)
        if tid not in index.keys():
            index[tid] = [entry]
        else:
            index[tid].append(entry)
    return index

def temporal_match_to_merra2_ds(tid,m2ds):
    fine_match = ps.cmp_temporal(np.array([tid],dtype=np.int64),merra2_stare_time(m2ds))
    return fine_match

def temporal_match_to_merra2(tid,m2_tid_index,dataPath=""):

    gm2_match = ps.cmp_temporal(np.array([tid],dtype=np.int64),list(m2_tid_index.keys())) # TODO Could speed up this using sortedcontainers? Would have to map to a particular level?

    match_fnames = []
    for i in range(gm2_match.size):
        if gm2_match[i] == 1:
            fine_match = temporal_match_to_merra2_ds(tid,Dataset(dataPath+m2_tid_index[list(m2_tid_index.keys())[i]][0]))
            if 1 in fine_match:
                match_fnames.append(m2_tid_index[list(m2_tid_index.keys())[i]][0])
            else:
                match_fnames.append(None)
        else:
            match_fnames.append(None)
    # print(entry, ' entry,matches: ',gm2_match,match_fnames)
    match_fnames_trimmed = []
    for i in match_fnames:
        if i is not None:
            match_fnames_trimmed.append(i)
    # print(entry,match_fnames_trimmed)
    if(len(match_fnames_trimmed) > 1):
        print('*** WARNING: more than one MERRA-2 file for the input tid file!!')
    return match_fnames_trimmed

def spatial_resolution(sid):
    return sid & 31 # levelMaskSciDB

def spatial_terminator_mask(level):
    return ((1 << (1+ 58-2*level))-1)

def spatial_terminator(sid):
    return sid | ((1 << (1+ 58-2*(sid & 31)))-1)

def spatial_coerce_resolution(sid,resolution):
    return (sid & ~31) | resolution

def spatial_clear_to_resolution(sid):
    resolution = sid & 31
    mask =  spatial_terminator_mask(spatial_resolution(sid))
    return (sid & ~mask) + resolution

###########################################################################

def simple_collect(sids,data,force_resolution=None):
    "Collect the data indexed by sids to an ROI indexed by sids."
    if force_resolution is None:
        sids_at_res = list(map(spatial_clear_to_resolution,sids))
    else:
        sids_at_res = [spatial_clear_to_resolution(spatial_coerce_resolution(s,force_resolution)) for s in sids]
    data_accum = dict()
    for s in sids_at_res:
        data_accum[s] = []
    for ics in range(len(sids_at_res)):
        data_accum[sids_at_res[ics]].append(data[ics])
    for cs in data_accum.keys():
        if len(data_accum[cs]) > 1:
            data_accum[cs] = [sum(data_accum[cs])/(1.0*len(data_accum[cs]))]
    tmp = np.array(list(data_accum.values()))
    vmin = np.amin(tmp)
    vmax = np.amax(tmp)
    return data_accum,vmin,vmax

###########################################################################

class modis_filename(object):
    def __init__(self,fn):
        fn_split = split(fn,'.')
        self.base_name = fn_split[0]
        self.adate     = fn_split[1]
        self.time      = fn_split[2]
        self.proc_id   = fn_split[3]
        self.proc_date = fn_split[4]
        self.ext       = fn_split[5]
        return
    def datetime(self):
        return '.'.join([self.adate,self.time])

###########################################################################

class data_catalog(object):
    def __init__(self,config):
        self.config             = config
        self.files              = None
        self.tid_centered_index = None
        return

    def get_files(self):
        if self.files is None:
            self.files = []
            if 'directory' in self.config.keys():
                dir = self.config['directory']
            else:
                dir = "./"
            filelist = os.listdir(dir)
            if 'patterns' in self.config.keys():
                patterns = self.config['patterns']
            else:
                patterns = ['*']
            for pattern in patterns:
                for entry in filelist:
                    if fnmatch.fnmatch(entry,pattern):
                        self.files.append(entry)        
        return self.files

    def get_tid_centered_index(self):
        if self.tid_centered_index == None:
            self.tid_centered_index = temporal_id_centered_filename_index(self.get_files())
        return self.tid_centered_index

    def find(self,tid):
        ok = False
        for p in self.config['patterns']:
            ok = ok or "MERRA" in p
        if ok:
            return temporal_match_to_merra2(tid
                                            ,self.get_tid_centered_index()
                                            ,dataPath=self.config['directory'])
        else:
            print('*ERROR* data_catalog.find not implemented for ',self.config['patterns'])
        return []

def hex16(i):
    return "0x%016x"%i

# def data_catalog_from_yaml(filename):
#     with open(filename) as f:
#         return data_catalog(yaml.load(f,Loader=yaml.FullLoader))
#     return None

# def make_hull(lat0,lon0,resolution0,ntri0):
#     hull0 = ps.to_hull_range_from_latlon(lat0,lon0,resolution0,ntri0)
#     lath0,lonh0,lathc0,lonhc0 = ps.to_vertices_latlon(hull0)
#     lons0,lats0,intmat0 = triangulate1(lath0,lonh0)
#     triang0 = tri.Triangulation(lons0,lats0,intmat0)
#     return lats0,lons0,triang0,hull0

###########################################################################
#
# HDFEOS & MODIS SUPPORT
#

###########################################################################
# https://gis.stackexchange.com/questions/328535/opening-eos-netcdf4-hdf5-file-with-correct-format-using-xarray
#
def parse_hdfeos_metadata(string):
  "Parse an extracted HDFEOS metadata string."
#  print('*********************************')
#  print('string: ',string)
#  print('')
  out = OrderedDict()
  lines0 = [i.replace('\t','') for i in string.split('\n')]
  lines = []
  for l in lines0:
      if "=" in l:
          key,value = l.split('=')
          lines.append(key.strip()+'='+value.strip())
      else:
          lines.append(l)

  i = -1
  while i<(len(lines))-1:
      i+=1
      line = lines[i]
      if "=" in line:
          key,value = line.split('=')
#          print('key: "%s"'%key)
          if key in ['GROUP','OBJECT']:
              endIdx = lines.index('END_{}={}'.format(key,value))
              out[value] = parse_hdfeos_metadata("\n".join(lines[i+1:endIdx]))
              i = endIdx
          else:
#              print('.')
              if ('END_GROUP' not in key) and ('END_OBJECT' not in key):
                  out[key] = str(value)
                  # try:
                  #     out[key] = eval(value)
                  # except NameError:
                  #     out[key] = str(value)
  return out
########

def with_hdf_get(h,var):
    "Select off and get a var, for convenience."
    sds = hdf.select(var)
    ret = sds.get()
    sds.endaccess()
    return ret

def modis_cover_from_gring(h,resolution=7,ntri_max=1000):
    "Read ArchiveMetadata.0 from file and extract GRING, creating STARE spatial cover."
    archive_metadata = h.attributes()['ArchiveMetadata.0']
    metadata = parse_hdfeos_metadata(archive_metadata)
    gring_seq=np.array(eval(metadata['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTSEQUENCENO']['VALUE'])[:],dtype=np.int)-1
    gring_lon=np.array(eval(metadata['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTLONGITUDE']['VALUE'])[:],dtype=np.double)
    gring_lat=np.array(eval(metadata['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTLATITUDE']['VALUE'])[:],dtype=np.double)
    return ps.to_hull_range_from_latlon(gring_lat[gring_seq],gring_lon[gring_seq],resolution,ntri_max)

if __name__ == '__main__':

  print('running')

# if True:
#   unittest.main()
        
