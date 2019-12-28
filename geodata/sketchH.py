#!/usr/bin/env python

import os

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

import numpy as np

import geodata as gd
import h5py as h5
from pyhdf.SD import SD, SDC

from stopwatch import sw_timer as timer

import pystare as ps

from sortedcontainers import SortedDict

###########################################################################
#
def safe_shape(x):
    try:
        ret = x.shape
    except:
        ret = None
        pass
    return ret
#
###########################################################################
#

class sare_partition(object):
    sid        = None
    data       = None
    dtype      = None
    name_base  = None
    var_name   = None
    shape      = None # (nAcross,nAlong)
    
    def __init__(self,sid,name_base=""):
        self.sid       = sid # The spatial id associated with the SARE partition
        self.name_base = name_base
        self.fname     = "%016x.%s.h5"%(sid,name_base)
        return

    def write1(self,shape=None,dataset_name="vars",vars={}):
        """
        Input
          shape - the shape of the original source array [nacross, nalong]
          vars - is a dictionary of np vars
            'sare' - spatial ids, a numpy array with a position for each row in the 'tables'
            'src_coord' - the index position in the original source array
        """
        self.dtype     = [(i,vars[i].dtype) for i in vars]
        # if sare and src_coord are not in self.dtype.names then raise?
        vars_ns  = np.array([len(vars[i]) for i in vars],dtype=np.int)
        vars_nmn = np.amin(vars_ns)
        vars_nmx = np.amax(vars_ns)
        if vars_nmn != vars_nmx:
            raise ValueError('Arrays to be output have different lengths.')
        vars_n = vars_nmn
        outFile = h5.File(self.fname,'w')

        outFile.create_dataset('metadata',[]
                               ,dtype=np.dtype([('sare_id',np.int64)
                                                ,('shape0',np.int32)
                                                ,('shape1',np.int32)
                                                ,('dataset_name','S%i'%len(dataset_name))
                               ]))
        outFile['metadata']['sare_id']      = self.sid
        outFile['metadata']['shape0']       = shape[0]
        outFile['metadata']['shape1']       = shape[1]
        outFile['metadata']['dataset_name'] = dataset_name
        
        outFile.create_dataset(dataset_name,[vars_n],dtype=self.dtype)
        for i in vars:
            outFile[dataset_name][i] = vars[i]
            
        outFile.close()
        return

    def read1(self):
        inFile = h5.File(self.fname,'r')

        # print('i.attrs: ',inFile.keys())
        dataset_name = inFile['metadata']['dataset_name']
        shape        = (inFile['metadata']['shape0'],inFile['metadata']['shape1'])
        sare         = inFile[dataset_name]['sare']
        src_coord    = inFile[dataset_name]['src_coord']
        # print('if.dtype: ',inFile[dataset_name].dtype)
        print('if.dtype.names: ',inFile[dataset_name].dtype.names)
        vars = {}
        for i in inFile[dataset_name].dtype.names:
            vars[i] = inFile[dataset_name][i].copy()
        inFile.close()
        return (shape,dataset_name,vars)
    
#
###########################################################################
#




#
###########################################################################
#
class modis05_set(object):
    data        = None
    locations   = None
    geo_latlon  = None
    geo_lat     = None
    geo_lon     = None
    sare        = None
    tare        = None
    data_wv_nir = None
    cover       = None
    nAlong      = None
    nAcross     = None
    def __init__(self,data,location,data_sourcedir=None,location_sourcedir=None):
        self.data     = data
        self.location = location
        self.data_sourcedir = data_sourcedir
        if location_sourcedir is None:
            location_sourcedir = data_sourcedir
        self.location_sourcedir = location_sourcedir
        return
    def load_geo(self):
        if self.geo_latlon is None:
            geo          = SD(self.location_sourcedir+self.location,SDC.READ)
            self.geo_lat = geo.select('Latitude').get()
            self.geo_lon = geo.select('Longitude').get()
            self.geo_latlon = (self.geo_lat,self.geo_lon)
            geo.end()
        return self
    def load_wv_nir(self):
        hdf              = SD(self.data_sourcedir+self.data,SDC.READ)
        if self.data_wv_nir is None:
            ds_wv_nir        = hdf.select('Water_Vapor_Near_Infrared')
            key_across       = 'Cell_Across_Swath_1km:mod05'
            key_along        = 'Cell_Along_Swath_1km:mod05'
            self.nAlong      = ds_wv_nir.dimensions()[key_along]
            self.nAcross     = ds_wv_nir.dimensions()[key_across]
            add_offset       = ds_wv_nir.attributes()['add_offset']
            scale_factor     = ds_wv_nir.attributes()['scale_factor']
            self.data_wv_nir = (ds_wv_nir.get()-add_offset)*scale_factor
            ds_wv_nir.endaccess()
            self.cover = gd.modis_cover_from_gring(hdf)
            hdf.end()
        return self
    def vmin(self):
        return np.amin(self.data_wv_nir)
    def vmax(self):
        return np.amax(self.data_wv_nir)
    def make_sare(self,res_km=1):
        self.sare = ps.from_latlon(self.geo_lat.flatten(),self.geo_lon.flatten(),int(gd.resolution(res_km)))
        return self
    def info(self):
        return '\n<modis05_set>' \
            +'\ndata              =%s'%self.data \
            +'\nlocation          =%s'%self.location \
            +'\ndata_wv_nir shape =%s'%str(safe_shape(self.data_wv_nir)) \
            +'\ngeo_latlon  shape =%s'%str(safe_shape(self.geo_lat)) \
            +'\nalong,across      =(%s, %s)'%(self.nAlong,self.nAcross)\
            +'\n</modis05_set>' \
            +'\n'
    #def companion(self):
    #    """Save translation & lookup tables."""
    #    return

def main():
    print('MODIS Sketching')

    data_sourcedir = '/home/mrilee/data/MODIS/'

    # Geolocation files
    mod03_catalog = gd.data_catalog({'directory':data_sourcedir
                                     ,'patterns':['MOD03*']})

    # Data files
    mod05_catalog = gd.data_catalog({'directory':data_sourcedir
                                     ,'patterns':['MOD05*']})

    print('mod03 catalog\n',mod03_catalog.get_files())
    print('mod03 catalog\n',mod03_catalog.get_tid_centered_index())
    print('mod05 catalog\n',mod05_catalog.get_tid_centered_index())

    modis_sets = SortedDict()
    tKeys = list(mod05_catalog.tid_centered_index.keys())
    for tid in tKeys[0:1]:
    # for tid in mod05_catalog.tid_centered_index: # replace with temporal comparison
        if(len(mod05_catalog.tid_centered_index[tid])>1 or len(mod03_catalog.tid_centered_index[tid])>1):
            raise NotImplementedError('sketchH only written for preselected pairs of MODIS files')
        modis_sets[tid] = modis05_set(mod05_catalog.tid_centered_index[tid][0]
                                      ,mod03_catalog.tid_centered_index[tid][0]
                                      ,data_sourcedir = data_sourcedir)
        modis_sets[tid].load_wv_nir().load_geo().make_sare()
        print(hex(tid),modis_sets[tid].data,modis_sets[tid].location)
        print(modis_sets[tid].info())

    ###########################################################################
    proj=ccrs.PlateCarree()
    transf = ccrs.Geodetic()

    def init_figure(proj):
        plt.figure()
        ax = plt.axes(projection=proj)
        ax.set_global()
        ax.coastlines()
        return ax

    vmin = np.amin(np.array([a.vmin() for a in modis_sets.values()]))
    vmax = np.amax(np.array([a.vmax() for a in modis_sets.values()]))
    tKeys = list(modis_sets.keys())
    tid   = tKeys[0]
    # tKeys = tKeys[1:]
    # tKeys = tKeys[-2:-1]
    # for tid in mod05_catalog.tid_centered_index: # replace with temporal comparison
    if True:
    # for tid in tKeys:
        if False:
            plt.scatter(
                modis_sets[tid].geo_lon
                ,modis_sets[tid].geo_lat
                ,s=1
                ,c=modis_sets[tid].data_wv_nir
                ,transform=transf
                ,vmin=vmin
                ,vmax=vmax
            )
        # clons,clats,cintmat = ps.triangulate_indices(modis_sets[tid].cover)
        tmp_cover = ps.to_compressed_range(modis_sets[tid].cover)
        # tmp_cover = ps.expand_intervals(tmp_cover,2)
        tmp_cover = ps.expand_intervals(tmp_cover,5)
        clons,clats,cintmat = ps.triangulate_indices(tmp_cover)
        ctriang = tri.Triangulation(clons,clats,cintmat)
        for sid in tmp_cover[0:1]:
            print(sid,' sid,cover sid: 0x%016x'%sid)
            idx = np.where(ps.cmp_spatial(np.array([sid],dtype=np.int64),modis_sets[tid].sare) != 0)
            spart_h5 = sare_partition(sid,'sketchH')
            spart_h5.write1(
                shape        = [modis_sets[tid].nAcross,modis_sets[tid].nAlong]
                ,dataset_name = 'wv_nir'
                ,vars         = {
                    'sare':modis_sets[tid].sare[idx]
                    ,'src_coord':np.arange(len(modis_sets[tid].sare))[idx]
                    ,'Water_Vapor_Near_Infrared':modis_sets[tid].data_wv_nir.flatten()[idx]}
                )
        if False:
            ax = init_figure(proj)
            ax.triplot(ctriang,'b-',transform=transf,lw=1.0,markersize=3,alpha=0.5)
            plt.scatter(
                modis_sets[tid].geo_lon.flatten()[idx]
                ,modis_sets[tid].geo_lat.flatten()[idx]
                ,s=1
                ,c=modis_sets[tid].data_wv_nir.flatten()[idx]
                ,transform=transf
                ,vmin=vmin
                ,vmax=vmax
            )
            plt.show()
        spart_h5_1 = sare_partition(sid,'sketchH')
        (s5_shape,s5_name,s5_vars) = spart_h5_1.read1()
        idx = s5_vars['src_coord']
        if False:
            ax = init_figure(proj)
            ax.triplot(ctriang,'g-',transform=transf,lw=1.0,markersize=3,alpha=0.5)
            plt.scatter(
                modis_sets[tid].geo_lon.flatten()[idx]
                ,modis_sets[tid].geo_lat.flatten()[idx]
                ,s=1
                ,c=modis_sets[tid].data_wv_nir.flatten()[idx]
                ,transform=transf
                ,vmin=vmin
                ,vmax=vmax
            )
            plt.show()
        if True:
            ax = init_figure(proj)
            ax.triplot(ctriang,'r-',transform=transf,lw=1.0,markersize=3,alpha=0.5)
            lat,lon = ps.to_latlon(s5_vars['sare'])
            plt.scatter(
                lon
                ,lat
                ,s=1
                ,c=s5_vars['Water_Vapor_Near_Infrared']
                ,transform=transf
                ,vmin=vmin
                ,vmax=vmax
            )
            plt.show()

    print('MODIS Sketching Done')
    return

if __name__ == '__main__':
    main()

