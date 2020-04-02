#!/usr/bin/env python


###########################################################################
# Partition MODIS granule into source files in STARE
###########################################################################

import os, fnmatch

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
    src_name   = None
    
    def __init__(self,sid,name_base="",src_name='None',var_nmax=None):
        self.sid       = sid # The spatial id associated with the SARE partition
        self.name_base = name_base
        self.fname     = "s%016x.%s.h5"%(sid,name_base)
        self.src_name  = src_name
        self.var_nmax  = var_nmax
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
        if self.var_nmax is not None:
            if self.var_nmax < vars_n:
                raise ValueError('var_nmax=%i is too small, vars_n=%i'%(self.var_nmax,vars_n))
            src_vars_n = vars_n
            tgt_vars_n = self.var_nmax
        else:
            src_vars_n = vars_n
            tgt_vars_n = vars_n

        # print('writing %i items into an array of %i items.'%(src_vars_n,tgt_vars_n))
        # print('input array size %i'%(len(vars['sare'])))

        outFile = h5.File(self.fname,'w')

        outFile.create_dataset('metadata',[1]
                               ,dtype=np.dtype([('sare_id',np.int64)
                                                ,('shape0',np.int32)
                                                ,('shape1',np.int32)
                                                ,('dataset_name','S1024')
                                                ,('src_name','S1024')
                                                # ,('dataset_name','S%i'%len(dataset_name))
                                                # ,('src_name','S%i'%len(self.src_name))
                                                ,('n_data',np.int64)
                               ]))
        outFile['metadata']['sare_id']       = self.sid
        outFile['metadata']['shape0']        = shape[0]
        outFile['metadata']['shape1']        = shape[1]
        outFile['metadata']['dataset_name']  = dataset_name
        outFile['metadata']['src_name']      = self.src_name
        outFile['metadata']['n_data']        = src_vars_n
        
        outFile.create_dataset(dataset_name,[tgt_vars_n],dtype=self.dtype)
        for i in vars:
            # outFile[dataset_name][i] = np.full([tgt_vars_n],-1,dtype=vars[i].dtype)
            # outFile[dataset_name][i][0:src_vars_n] = vars[i][0:src_vars_n].copy()
            tmp = np.full([tgt_vars_n],-1,dtype=vars[i].dtype)
            tmp[0:src_vars_n] = vars[i][0:src_vars_n]
            outFile[dataset_name][i] = tmp
            ## if src_vars_n > 0:
            ##     print(i,' 200 i,mn,mx ',np.amin(tmp),np.amax(tmp))
            ##     print(i,' 200 i,mn,mx ',np.amin(vars[i]),np.amax(vars[i]))
            ## else:
            ##     print(i,' 201 i,mn,mx ',vars[i])
            ##     print(i,' 201 i,mn,mx ',tmp)
            # print(i,' i,tmp ',tmp)
            
        outFile.close()
        return

    def read1(self):
        inFile = h5.File(self.fname,'r')

        # print('i.attrs: ',inFile.keys())
        dataset_name = inFile['metadata']['dataset_name'][0]
        shape        = (inFile['metadata']['shape0'][0],inFile['metadata']['shape1'][0])
        src_vars_n   = inFile['metadata']['n_data'][0]
        # print('read1 reading %s of shape %s and %i records.'%(dataset_name,shape,src_vars_n))
        sare         = inFile[dataset_name]['sare'][0:src_vars_n].copy()
        src_coord    = inFile[dataset_name]['src_coord'][0:src_vars_n].copy()
        # print('if.dtype: ',inFile[dataset_name].dtype)
        # print('if.dtype.names: ',inFile[dataset_name].dtype.names)
        # print('reading %i items each.'%(src_vars_n))
        vars = {}
        for i in inFile[dataset_name].dtype.names:
            vars[i] = np.zeros([src_vars_n],dtype=inFile[dataset_name][i].dtype)
            vars[i][:] = inFile[dataset_name][i][0:src_vars_n].copy()
            ## print('    %s: type '%(i),type(vars[i]),vars[i].dtype,vars[i].shape)
            ## if vars[i].size > 0:
            ##     print('    %s: mn,mx = %s,%s'%(i,np.amin(vars[i]),np.amax(vars[i])))
            ## else:
            ##     print('    %s: mn,mx = %s,%s'%(i,vars[i],vars[i]))
            # print(i,' i,vars[i] ',vars[i])
        vars_dtype     = inFile[dataset_name].dtype;
        metadata_dtype = inFile['metadata'].dtype;
        inFile.close()
        return (shape,dataset_name,vars,vars_dtype,metadata_dtype)
    
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
    for tid in tKeys:
    # for tid in tKeys[0:1]:
    # for tid in mod05_catalog.tid_centered_index: # replace with temporal comparison
        if(len(mod05_catalog.tid_centered_index[tid])>1 or len(mod03_catalog.tid_centered_index[tid])>1):
            raise NotImplementedError('sketchH only written for preselected pairs of MODIS files')
        modis_sets[tid] = modis05_set(mod05_catalog.tid_centered_index[tid][0]
                                      ,mod03_catalog.tid_centered_index[tid][0]
                                      ,data_sourcedir = data_sourcedir)
        modis_sets[tid].load_wv_nir().load_geo().make_sare()
        # print(hex(tid),modis_sets[tid].data,modis_sets[tid].location)
        # print(modis_sets[tid].info())

    ###########################################################################
    proj=ccrs.PlateCarree()
    transf = ccrs.Geodetic()

    def init_figure(proj):
        plt.figure()
        ax = plt.axes(projection=proj)
        # ax.set_global()
        # ax.coastlines()
        return ax

    vmin = np.amin(np.array([a.vmin() for a in modis_sets.values()]))
    vmax = np.amax(np.array([a.vmax() for a in modis_sets.values()]))
    tKeys = list(modis_sets.keys())
    tKeys = [tKeys[1]]
    # tid   = tKeys[0]
    # tKeys = tKeys[1:]
    # tKeys = tKeys[-2:-1]
    # for tid in mod05_catalog.tid_centered_index: # replace with temporal comparison
    # if True:
    for tid in tKeys:
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
        
        tmp_data_src_name = mod05_catalog.tid_centered_index[tid][0][0:-4]
        # print('tmp_data_src_name: ',tmp_data_src_name)
        # clons,clats,cintmat = ps.triangulate_indices(modis_sets[tid].cover)
        tmp_cover = ps.to_compressed_range(modis_sets[tid].cover)
        # tmp_cover = ps.expand_intervals(tmp_cover,2)
        tmp_cover = ps.expand_intervals(tmp_cover,4)
        # tmp_cover = ps.expand_intervals(tmp_cover,6,result_size_limit=1600)
        if False:
            clons,clats,cintmat = ps.triangulate_indices(tmp_cover)
            ctriang = tri.Triangulation(clons,clats,cintmat)
        spart_nmax = -1
        # tmp_cover = tmp_cover[0:10]
        idx_all = {}
        for sid in tmp_cover:
            # print(sid,' sid, indexing cover sid: 0x%016x'%sid)
            idx_all[sid] = np.where(ps.cmp_spatial(np.array([sid],dtype=np.int64),modis_sets[tid].sare) != 0)
            spart_nmax = max(spart_nmax,len(modis_sets[tid].sare[idx_all[sid]]))
        # print('max spart items to write per file: ',spart_nmax)
        spart_nmax = None # Variable lengths
        
        spart_names = []
        for sid in tmp_cover:
            idx = idx_all[sid][0]
            # print(sid,' sid,cover sid: 0x%016x'%sid,' len(idx)=%i'%(len(idx)))
            # print('idx:      ',idx)
            # print('idx type: ',type(idx))
            spart_h5_namebase = 't%016x.%s'%(tid,tmp_data_src_name+'.sketchJ0')

            # MAKE
            spart_h5 = sare_partition(sid,spart_h5_namebase,src_name=tmp_data_src_name,var_nmax = spart_nmax)

            # CHECK
            # print(' data_wv_nir mn,mx: ',np.amin(modis_sets[tid].data_wv_nir),np.amax(modis_sets[tid].data_wv_nir))

            # WRITE
            if True:
                spart_h5.write1(
                    shape        = [modis_sets[tid].nAcross,modis_sets[tid].nAlong]
                    ,dataset_name = 'wv_nir'
                    ,vars         = {
                        'sare':modis_sets[tid].sare[idx]
                        ,'src_coord':np.arange(len(modis_sets[tid].sare))[idx]
                        ,'Water_Vapor_Near_Infrared':modis_sets[tid].data_wv_nir.flatten()[idx]}
                )
                # print('')
            
            spart_names.append(spart_h5.fname)
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

            # READ
            spart_h5_1 = sare_partition(sid,spart_h5_namebase)
            (s5_shape,s5_name,s5_vars,s5_vars_dtype,s5_metadata_dtype) = spart_h5_1.read1()
            # print('found and loaded %s of type %s.'%(s5_name,s5_vars_dtype))
            
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
            if False:
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

        ####

        ## # Define the layout of the virtual data set.
        ## var_ns      = [len(idx_all[i][0]) for i in idx_all]
        ## var_n_cumul = [0]
        ## for i in range(len(idx_all)):
        ##     var_n_cumul.append(var_n_cumul[-1]+var_ns[i])
        ## # var_n_total = sum([len(i) for i in idx_all])
        ## var_n_total = var_n_cumul[-1]
        ## 
        ## print('')
        ## print('var_ns:      ',var_ns)
        ## print('var_n_cumul: ',var_n_cumul)
        ## print('var_n_total: ',var_n_total)
        ## 
        ## layout    = h5.VirtualLayout(shape=(var_n_total,),dtype=s5_vars_dtype)
        ## print('layout:      ',layout)
        ## 
        ## layout_md = h5.VirtualLayout(shape=(len(spart_names),),dtype=s5_metadata_dtype)
        ## print('layout_md:   ',layout_md)
        ## print('')
        ## 
        ## layout_idx = np.arange(var_n_total)
        ## 
        ## for i in range(len(spart_names)):
        ##     print(i,' part: ',var_n_cumul[i],var_n_cumul[i+1],', ',var_ns[i])
        ## print('')
        ## 
        ## # print('s5_metadata_dtype: ',s5_metadata_dtype)
        ## ds_name = 'wv_nir'
        ## for i in range(len(spart_names)):
        ##     # Just concatenate
        ##     #++ layout[var_n_cumul[i]:var_n_cumul[i+1]] = h5.VirtualSource(spart_names[i],ds_name,shape=(var_ns[i],))
        ##     # What I would like to do...
        ##     if var_ns[i] != 0:
        ##         with h5.File(spart_names[i]) as h:
        ##             vs = h5.VirtualSource(spart_names[i],ds_name,shape=(var_ns[i],))
        ##             for j in range(h[ds_name]['src_coord'].shape[0]): # Should be var_ns[i]
        ##                 layout[h[ds_name]['src_coord'][j]] = vs[j] # next step, try [j,k] for a virtual image....
        ##     # else: # zero-case
        ##     #    layout[var_n_cumul[i]:var_n_cumul[i+1]] = h5.VirtualSource(spart_names[i],ds_name,shape=(var_ns[i],))                        
        ##     
        ## for i in range(len(spart_names)):
        ##     layout_md[i] = h5.VirtualSource(spart_names[i],'metadata',shape=(1,))
        ##     
        ## vds_fname = '.'.join([tmp_data_src_name,ds_name,'sketchJ0.vds.h5'])
        ## print('writing ',vds_fname)
        ## with h5.File(vds_fname,'w',libver='latest') as f:
        ##     f.create_virtual_dataset('wv_nir',layout)      # fillvalue?
        ##     f.create_virtual_dataset('metadata',layout_md) # fillvalue?
        ##     # f.create_dataset(['image_lookup']...)

    print('MODIS Sketching Done')
    return

if __name__ == '__main__':
    main()

