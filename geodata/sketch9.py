
import geodata as gd
import h5py as h5
from netCDF4 import Dataset
import numpy as np
import pystare as ps
import json
from sortedcontainers import SortedDict, SortedList

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

### GOES DATASET
goes_b5_dataPath = "/home/mrilee/data/"
goes_b5_dataFile = "goes10.2005.349.003015.BAND_05.nc"
goes_b5_fqFilename = goes_b5_dataPath+goes_b5_dataFile
goes_b5_ds = Dataset(goes_b5_fqFilename)

g5shape = goes_b5_ds['data'].shape
print('g5shape = ',g5shape)

g5size = goes_b5_ds['data'].size
print('g5size = ',g5size)

goes_b5_tid = gd.goes10_img_stare_time(goes_b5_ds)

### MERRA 2 DATASET
dataPath   = "/home/mrilee/data/"
dataFile   = "MERRA2_300.tavg1_2d_slv_Nx.20051215.nc4"
fqFilename = dataPath+dataFile
m2_dLat    = 0.5
m2_dLon    = 5.0/8.0
m2_dLatkm  = m2_dLat * gd.re_km/gd.deg_per_rad
m2_dLonkm  = m2_dLon * gd.re_km/gd.deg_per_rad
m2_ds      = Dataset(fqFilename)
print('dims lat,lon: ',m2_ds['lat'].shape,m2_ds['lon'].shape)
m2_lat,m2_lon = np.meshgrid(m2_ds['lat'],m2_ds['lon'])
m2_lat     = m2_lat.flatten()
m2_lon     = m2_lon.flatten()
m2_idx_ij  = np.arange(m2_lat.size,dtype=np.int64)
m2_resolution = int(gd.resolution(m2_dLonkm*2))
m2_indices = ps.from_latlon(m2_lat,m2_lon,m2_resolution)
# m2_indices = ps.from_latlon(m2_lat,m2_lon,int(gd.resolution(m2_dLonkm*0.5)))
m2_tid     = gd.merra2_stare_time(m2_ds)

print('type(m2_indices): ',type(m2_indices))
print('shape(m2_indices): ',m2_indices.shape)
print('size(m2_indices): ',m2_indices.size)
print('type(g5 tid): ',type(goes_b5_tid))
print('g5 tid.shape: ',goes_b5_tid.shape)
print('type(m2_tid): ',type(m2_tid))
print('m2_tid.shape: ',m2_tid.shape)
fine_match = ps.cmp_temporal(np.array(goes_b5_tid,dtype=np.int64),m2_tid)
m2_ifm     = np.nonzero(fine_match)[0]

m2_dataDayI     = m2_ds['TQI'][m2_ifm,:,:]
m2_dataDayL     = m2_ds['TQL'][m2_ifm,:,:]
m2_dataDayV     = m2_ds['TQV'][m2_ifm,:,:]
m2_dataDay      = m2_dataDayI + m2_dataDayL + m2_dataDayV
m2_data         = m2_dataDay[:,:].T
m2_data_flat    = m2_data.flatten()
print('m2 data mnmx: ',np.amin(m2_data_flat),np.amax(m2_data_flat))

### HDF5 SAVE DATASET
# workPath     = "/home/mrilee/tmp/"
workFileName = "work.h5"
#workFile     = h5.File(workPath+workFileName,'w')
workFile     = h5.File(workFileName,'w')

image_dtype = np.dtype([
    ('stare_spatial',np.int64)
    ,('stare_temporal',np.int64)
    ,('goes_src_coord',np.int64)
    ,('goes_b3',np.int64)
    ,('goes_b4',np.int64)
    ,('goes_b5',np.int64)
    ,('merra2_src_coord',np.int64)
    ,('merra2_tpw',np.int64)
])
image_ds = workFile.create_dataset('image',[g5size],dtype=image_dtype)

image_description_dtype = np.dtype([
    ('nx',np.int)
    ,('ny',np.int)
])
image_description_ds = workFile.create_dataset('image_description',[],dtype=image_description_dtype)
# image_description_ds['image_nx']=1024

m2_description_dtype = np.dtype([
    ('nx',np.int)
    ,('ny',np.int)
    ,('tpw_offset',np.double)
    ,('tpw_scale',np.double)
])
m2_description_ds = workFile.create_dataset('merra2_description',[],dtype=m2_description_dtype)

print("image_ds['goes_b5'].size: ",image_ds['goes_b5'].size)
print("image_description_ds['nx']: ",image_description_ds['nx'].size)
print("merra2_description_ds['merra2_description'].size: ",m2_description_ds['nx'].size)
print('')

g_lat = goes_b5_ds['lat'][:,:].flatten()
g_lon = goes_b5_ds['lon'][:,:].flatten()
print('g_lat.size  ',g_lat.size)
print('g_lat.shape ',g_lat.shape)
g_idx_valid = np.where((g_lat>=-90.0) & (g_lat<=90.0))
g_idx_invalid = np.where(((g_lat<-90.0) | (g_lat>90.0)))
# g_idx_ij = np.arange(g_lat.size,dtype=np.int64)

print('type(g_idx_valid):',type(g_idx_valid))
print('len(g_idx_valid): ',len(g_idx_valid))
# print('g_idx_valid:      ',g_idx_valid)
print('type(g_idx_valid[0]):',type(g_idx_valid[0]))
print('len(g_idx_valid[0]): ',len(g_idx_valid[0]))

goes_b5_indices = np.full(g_lat.shape,-1,dtype=np.int64)
goes_b5_indices[g_idx_valid] = ps.from_latlon(g_lat[g_idx_valid],g_lon[g_idx_valid],int(gd.resolution(goes_b5_ds['elemRes'][0])))
#-# goes_b5_indices[g_idx_invalid] = -1


### Join the M2 data
# print('len(g_idx_valid[0]): ',len(g_idx_valid[0]))

# workFile['/image']['merra2_src_coord'] = np.full(goes_b5_indices.shape,-1,dtype=np.int64)
# workFile['/image']['merra2_tpw']       = np.full(goes_b5_indices.shape,-1,dtype=np.int64)

print('calculating terminators')
m2_term = gd.spatial_terminator(m2_indices)

m2_src_coord_h5 = np.full(g_lat.shape,-1,dtype=np.int64)
m2_tpw_h5       = np.full(g_lat.shape,-1,dtype=np.int64)

class join_value(object):
    def __init__(self):
        self.bandmaps = {}
        return

    def contains(self,id):
        return id in self.bandmaps.keys()

    def add(self,bandname,id):
        if bandname not in self.bandmaps.keys():
            self.bandmaps[bandname] = SortedList()
        self.bandmaps[bandname].add(id)
        return

    def get(self,bandname):
        return self.bandmaps[bandname]

    def toJSON(self):
        str = ""
        output = {}
        for imap in self.bandmaps.keys():
            # print('saving ',imap)
            value = [int(i) for i in self.bandmaps[imap]]
            output[imap]=value
            # print('value:  ',value)
            # str = str + json.dumps( {imap:[i for i in self.bandmaps[imap]]} )
            # str = str + json.dumps( {imap:value} ) + '\n'
        str = json.dumps(output)+"\n"
        return str

def hex16(i):
    return "0x%016x"%i

join_resolution = m2_resolution
join = SortedDict()

ktr=0
print('add goes b5 to join')
for k in range(len(g_idx_valid[0])):
    id = g_idx_valid[0][k]
    jk = gd.spatial_clear_to_resolution(gd.spatial_coerce_resolution(goes_b5_indices[id],join_resolution))
    # print('id,jk: ',id,hex16(jk))
    # print('keys:  ',[hex16(i) for i in join.keys()])
    if jk not in join.keys():
        join[jk] = join_value()
    join[jk].add('goes_b5',id)
    ktr = ktr + 1; 
    # if ktr > 10:
    #     break
    #     # exit();

print('add MERRA-2 to join')
for k in range(len(m2_indices)):
    jk = gd.spatial_clear_to_resolution(m2_indices[k])
    if jk not in join.keys():
        join[jk] = join_value()
    join[jk].add('m2',k)

print('len(join): ',len(join))

# print('-----')
# for k in range(21000,21100):
#     print('--')
#     j = join.peekitem(k)
#     print(k,' join ',hex16(j[0]),'\n',j[1].toJSON(),'---')

# exit()

tpw_scale  = 0.001;
tpw_offset = 0;

print('----')
jkeys=join.keys()
ktr = 0; nktr = len(jkeys) # gd_idx_valid is a tuple with one element
dktr = nktr/10.0
elements_pushed = 0
print('Push joined m2 data into the dataset n = ',nktr)
for k in range(nktr):
    ktr = ktr + 1
    if (ktr % int(dktr)) == 0:
        print(int((10.0*ktr)/dktr),'% complete, ',elements_pushed,' elements pushed.')
    sid = jkeys[k]
    if join[sid].contains('goes_b5'):
        if join[sid].contains('m2'):
            m2s = join[sid].get('m2')[0] # Grab the first one
            m2_src_coord_h5[join[sid].get('goes_b5')] = m2s
            # m2_tpw_h5[join[sid].get('goes_b5')]       = (m2_data_flat[m2s]-tpw_offset)/tpw_scale
            avg = (np.mean(m2_data_flat[join[sid].get('m2')])-tpw_offset)/tpw_scale
            m2_tpw_h5[join[sid].get('goes_b5')]       = avg
            elements_pushed = elements_pushed + len(join[sid].get('goes_b5'))

print('m2_tpw_h5 shape: ',m2_tpw_h5.shape)
print('m2_tpw_h5 mnmx:  ',np.amin(m2_tpw_h5),np.amax(m2_tpw_h5))

workFile['/image']['stare_spatial'] = goes_b5_indices[:]
workFile['/image']['stare_temporal'] = gd.goes10_img_stare_time(goes_b5_ds)[0]
workFile['/image']['goes_src_coord'] = np.arange(g_lat.size,dtype=np.int64)
workFile['/image']['goes_b3'] = goes_b5_ds['data'][0,:,:].flatten()
workFile['/image']['goes_b4'] = goes_b5_ds['data'][0,:,:].flatten()
workFile['/image']['goes_b5'] = goes_b5_ds['data'][0,:,:].flatten()
workFile['/image']['merra2_src_coord'] = m2_src_coord_h5.flatten()
workFile['/image']['merra2_tpw']       = m2_tpw_h5.flatten()
workFile['/image_description']['nx'] = goes_b5_ds['data'].shape[1]
workFile['/image_description']['ny'] = goes_b5_ds['data'].shape[2]
workFile['/merra2_description']['nx'] = 576
workFile['/merra2_description']['ny'] = 361
workFile['/merra2_description']['tpw_offset'] = tpw_offset
workFile['/merra2_description']['tpw_scale']  = tpw_scale
workFile.close()

fig, (ax0,ax1) = plt.subplots(nrows=2)

nx = goes_b5_ds['data'].shape[1]
ny = goes_b5_ds['data'].shape[2]

b5_img = goes_b5_ds['data'][0,:,:].flatten().reshape(nx,ny)
print('b5 mnmx: ',np.amin(b5_img),np.amax(b5_img))
ax0.set_title('b5')
ax0.imshow(b5_img)

m2_img = m2_tpw_h5.reshape(nx,ny)
print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))
ax1.set_title('tpw')
ax1.imshow(m2_img)
plt.show()

