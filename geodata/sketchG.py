
# Load MODIS and construct an h5 file.

import numpy as np
from pyhdf.SD import SD, SDC
# import pprint

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

import geodata as gd
import pystare as ps
import h5py as h5

import modis_coarse_to_fine_geolocation.modis_5km_to_1km_geolocation as pascal_modis

dataPath="/home/mrilee/data/"

modis_base   = "MOD05_L2."

# 1 modis_item       = "A2005349.2120.061.2017294065852"
# 1 modis_time_start = "2005-12-15T21:20:00"

modis_item       = "A2005349.2125.061.2017294065400"
modis_time_start = "2005-12-15T21:25:00"

modis_suffix = ".hdf"
modis_filename = modis_base+modis_item+modis_suffix

fmt_suffix = ".h5"
workFileName = "sketchG."+modis_base+modis_item+fmt_suffix

key_across = 'Cell_Across_Swath_1km:mod05'
key_along  = 'Cell_Along_Swath_1km:mod05'

hdf        = SD(dataPath+modis_filename,SDC.READ)
ds_wv_nir  = hdf.select('Water_Vapor_Near_Infrared')
data       = ds_wv_nir.get()

add_offset   = ds_wv_nir.attributes()['add_offset']
scale_factor = ds_wv_nir.attributes()['scale_factor']
print('scale_factor = %f, add_offset = %f.'%(scale_factor,add_offset))
data = (data-add_offset)*scale_factor
print('data mnmx: ',np.amin(data),np.amax(data))

nAlong  = ds_wv_nir.dimensions()[key_along]
nAcross = ds_wv_nir.dimensions()[key_across]
print('ds_wv_nir nAlong,nAcross: ',nAlong,nAcross)

dt = np.array([modis_time_start],dtype='datetime64[ms]')
t_resolution = 26 # 5 minutes resolution? 2+6+10+6+6
tid = ps.from_utc(dt.astype(np.int64),t_resolution)
# print(np.arange(np.datetime64("2005-12-15T21:20:00"),np.datetime64("2005-12-15T21:25:00")))
# exit()

fill_value = ds_wv_nir.attributes()['_FillValue']

# print(ds_wv_nir.info())
# print(hdf.info())
# print(ds_wv_nir.attributes())
# print(hdf.attributes().keys())

# hdf_lat = hdf.select('Latitude')
# print('hdf.lat: ',type(hdf_lat),hdf_lat.info())

def with_hdf_get(h,var):
    sds = hdf.select(var)
    ret = sds.get()
    sds.endaccess()
    return ret

lat_5km = with_hdf_get(hdf,'Latitude')
lon_5km = with_hdf_get(hdf,'Longitude')
print('lat_5km ',type(lat_5km),lat_5km.dtype,lat_5km.shape)

# itk_1km: grid index along track
# isc_1km: grid index across track

lat = np.zeros([nAlong,nAcross],dtype=np.double)
lon = np.zeros([nAlong,nAcross],dtype=np.double)
stare_spatial  = np.zeros([nAlong,nAcross],dtype=np.int64)
stare_temporal = np.zeros([nAlong,nAcross],dtype=np.int64)

resolution = int(gd.resolution(2)) # km
print('resolution ',type(resolution))
ktr = 0
ktr_max = nAlong*nAcross
print('interpolate from 5km to 1km: %2d%%'%int(100*ktr/ktr_max),end='\r',flush=True)
for itk_1km in range(nAlong):
    for isc_1km in range(nAcross):
        if int(100*ktr/ktr_max) % 5 == 0 or int(100*ktr/ktr_max) < 2:
            print('interpolate from 5km to 1km: %2d%%'%int(100*ktr/ktr_max),end='\r',flush=True)
        lat[itk_1km,isc_1km],lon[itk_1km,isc_1km] = pascal_modis.get_1km_pix_pos(itk_1km,isc_1km,lat_5km,lon_5km)
        stare_spatial[itk_1km,isc_1km]  = ps.from_latlon(
            np.array([lat[itk_1km,isc_1km]])
            ,np.array([lon[itk_1km,isc_1km]])
            ,resolution)
        stare_temporal[itk_1km,isc_1km] = tid # TODO use np.timedelta for more temporal resolution
        ktr = ktr + 1
print('interpolate from 5km to 1km: done')

workFile = h5.File(workFileName,'w')

image_dtype = np.dtype([
     ('stare_spatial',np.int64)
    ,('stare_temporal_start',np.int64)
    ,('src_coord',np.int64)
    ,('Latitude',np.double)
    ,('Longitude',np.double)
    ,('Water_Vapor_Near_Infrared',np.double)
])
image_ds = workFile.create_dataset('image',[nAlong*nAcross],dtype=image_dtype)

image_description_dtype = np.dtype([
    ('src_file','S80')
    ,('nAlong',np.int)
    ,('nAcross',np.int)
])
image_description_ds = workFile.create_dataset('image_description',[],dtype=image_description_dtype)

workFile['/image']['stare_spatial']             = stare_spatial[:,:].flatten()
workFile['/image']['stare_temporal_start']      = stare_temporal[:,:].flatten()
workFile['/image']['src_coord']                 = np.arange(nAlong*nAcross,dtype=np.int64)
workFile['/image']['Latitude']                  = lat[:,:].flatten()
workFile['/image']['Longitude']                 = lon[:,:].flatten()
workFile['/image']['Water_Vapor_Near_Infrared'] = data[:,:].flatten()
workFile['/image_description']['src_file']      = modis_filename.encode("ascii","ignore")
workFile['/image_description']['nAlong']        = nAlong
workFile['/image_description']['nAcross']       = nAcross
# print("wf: '%s'"%workFile['/image_description']['src_file'])
workFile.close()   

proj=ccrs.PlateCarree()
transf = ccrs.Geodetic()

plt.figure()
ax = plt.axes(projection=proj)
ax.set_global()
ax.coastlines()
plt.scatter(lon,lat,s=1,c=data,transform=transf)
plt.show()

"""
http://www.icare.univ-lille1.fr/wiki/index.php/MODIS_geolocation
"""

"""
short Water_Vapor_Near_Infrared(Cell_Along_Swath_1km=2030, Cell_Across_Swath_1km=1354);
  :long_name = "Total Column Precipitable Water Vapor - Near Infrared Retrieval";
  :unit = "cm";
  :scale_factor = 0.0010000000474974513; // double
  :add_offset = 0.0; // double
  :Parameter_Type = "Output";
  :Cell_Along_Swath_Sampling = 1, 2030, 1; // int
  :Cell_Across_Swath_Sampling = 1, 1354, 1; // int
  :Geolocation_Pointer = "Internal geolocation arrays";
  :valid_range = 0S, 20000S; // short
  :_FillValue = -9999S; // short
"""

hdf.end()
