
# Load and check sketch5.py's output. Load the source file and try the GRING.

import numpy as np

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

import geodata as gd
import pystare as ps
import h5py as h5
from pyhdf.SD import SD, SDC


dataPath="/home/mrilee/data/"

modis_base   = "MOD05_L2."

# modis_item   = "A2005349.2120.061.2017294065852"
# modis_time_start = "2005-12-15T21:20:00"

modis_item       = "A2005349.2125.061.2017294065400"
modis_time_start = "2005-12-15T21:25:00"

modis_suffix = ".hdf"
modis_filename = modis_base+modis_item+modis_suffix

hdf        = SD(dataPath+modis_filename,SDC.READ)
ds_wv_nir  = hdf.select('Water_Vapor_Near_Infrared')

fmt_suffix = ".h5"
workFileName = "sketchG."+modis_base+modis_item+fmt_suffix
workFile = h5.File(workFileName,'r')
lat  = workFile['/image']['Latitude']
lon  = workFile['/image']['Longitude']
data = workFile['/image']['Water_Vapor_Near_Infrared']
workFile.close()

## ok ## # print('hdf att ',hdf.attributes().keys())
## ok ## # print('hdf arc ',hdf.attributes()['ArchiveMetadata.0'])
## ok ## archive_metadata = hdf.attributes()['ArchiveMetadata.0']
## ok ## # print('archive_metadata ',archive_metadata)
## ok ## # print('hdf att ',hdf.attributes()['GRING'])
## ok ## 
## ok ## metadata_parsed = gd.parse_hdfeos_metadata(archive_metadata)
## ok ## # print('metadata ',metadata_parsed)
## ok ## # k=0
## ok ## # for i in metadata_parsed:
## ok ## #     print(k,i,metadata_parsed[i])
## ok ## #     k=k+1
## ok ## 
## ok ## # print(metadata_parsed['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER'].keys())
## ok ## # print(metadata_parsed['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT'].keys())
## ok ## # print(metadata_parsed['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTLONGITUDE'].keys())
## ok ## # print(eval(metadata_parsed['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTLONGITUDE']['VALUE'])[:])
## ok ## # print(eval(metadata_parsed['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTLATITUDE']['VALUE'])[:])
## ok ## # print(eval(metadata_parsed['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTSEQUENCENO']['VALUE'])[:])
## ok ## # hdf.end()
## ok ## 
## ok ## gring_seq=np.array(eval(metadata_parsed['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTSEQUENCENO']['VALUE'])[:],dtype=np.int)-1
## ok ## gring_lon=np.array(eval(metadata_parsed['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTLONGITUDE']['VALUE'])[:],dtype=np.double)
## ok ## gring_lat=np.array(eval(metadata_parsed['ARCHIVEDMETADATA']['GPOLYGON']['GPOLYGONCONTAINER']['GRINGPOINT']['GRINGPOINTLATITUDE']['VALUE'])[:],dtype=np.double)
## ok ## 
resolution = 7
ntri_max   = 1000
# hull = ps.to_hull_range_from_latlon(gring_lat[gring_seq],gring_lon[gring_seq],resolution,ntri_max)
hull = gd.modis_cover_from_gring(hdf,resolution,ntri_max)
hdf.end()

lons,lats,intmat = ps.triangulate_indices(hull)
triang = tri.Triangulation(lons,lats,intmat)

proj   = ccrs.PlateCarree()
# proj   = ccrs.Mollweide()
# proj   = ccrs.Mollweide(central_longitude=-160.0)
transf = ccrs.Geodetic()

plt.figure()
ax = plt.axes(projection=proj)
ax.set_title('G-RING Test')
ax.set_global()
ax.coastlines()
if True:
    plt.scatter(lon,lat,s=1,c=data,transform=transf)
plt.triplot(triang,'b-',transform=transf,lw=1,markersize=2)

crop_lat=(  27.0,  39.0)
crop_lon=(-161.5,-159.5)
crop_lats = np.array([crop_lat[0],crop_lat[0],crop_lat[1],crop_lat[1]],dtype=np.double)
crop_lons = np.array([crop_lon[0],crop_lon[1],crop_lon[1],crop_lon[0]],dtype=np.double)
ar_resolution = 7
ar_cover = ps.to_hull_range_from_latlon(crop_lats,crop_lons,ar_resolution)
# print('ar_cover size: ',ar_cover.size)
# ar_cover_mn = np.amin(ar_cover)
# ar_cover_mx = np.amax(ar_cover)
lons,lats,intmat = ps.triangulate_indices(ar_cover)
triang = tri.Triangulation(lons,lats,intmat)
plt.triplot(triang,'r-',transform=transf,lw=1.5,markersize=3)

plt.show()

