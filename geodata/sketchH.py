#!/usr/bin/env python

import os

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

import numpy as np

import geodata as gd
from pyhdf.SD import SD, SDC

def safe_shape(x):
    try:
        ret = x.shape
    except:
        ret = None
        pass
    return ret

class modis05_set(object):
    data        = None
    locations   = None
    geo_latlon  = None
    geo_lat     = None
    geo_lon     = None
    data_wv_nir = None
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
            hdf.end()
        return self
    def vmin(self):
        return np.amin(self.data_wv_nir)
    def vmax(self):
        return np.amax(self.data_wv_nir)
    def info(self):
        return '\n<modis05_set>' \
            +'\ndata              =%s'%self.data \
            +'\nlocation          =%s'%self.location \
            +'\ndata_wv_nir shape =%s'%str(safe_shape(self.data_wv_nir)) \
            +'\ngeo_latlon  shape =%s'%str(safe_shape(self.geo_lat)) \
            +'\nalong,across      =(%s, %s)'%(self.nAlong,self.nAcross)\
            +'\n</modis05_set>' \
            +'\n'




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

    modis_sets = dict()
    for tid in mod05_catalog.tid_centered_index: # replace with temporal comparison
        if(len(mod05_catalog.tid_centered_index[tid])>1 or len(mod03_catalog.tid_centered_index[tid])>1):
            raise NotImplementedError('sketchH only written for preselected pairs of MODIS files')
        modis_sets[tid] = modis05_set(mod05_catalog.tid_centered_index[tid][0]
                                      ,mod03_catalog.tid_centered_index[tid][0]
                                      ,data_sourcedir = data_sourcedir)
        modis_sets[tid].load_wv_nir().load_geo()
        print(hex(tid),modis_sets[tid].data,modis_sets[tid].location)
        print(modis_sets[tid].info())
        

    ###########################################################################
    proj=ccrs.PlateCarree()
    transf = ccrs.Geodetic()
    
    plt.figure()
    
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.coastlines()

    vmin = np.amin(np.array([a.vmin() for a in modis_sets.values()]))
    vmax = np.amax(np.array([a.vmax() for a in modis_sets.values()]))
    for tid in mod05_catalog.tid_centered_index: # replace with temporal comparison
        plt.scatter(
            modis_sets[tid].geo_lon
            ,modis_sets[tid].geo_lat
            ,s=1
            ,c=modis_sets[tid].data_wv_nir
            ,transform=transf
            ,vmin=vmin
            ,vmax=vmax
        )
                    
    plt.show()

    print('MODIS Sketching Done')
    return

if __name__ == '__main__':
    main()

