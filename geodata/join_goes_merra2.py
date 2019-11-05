
# geodata/join_goes_merra2.py

# Read GOES and MERRA-2 files, spatiotemporally join bands 3, 4, and 5 with TPQ. Write to hdf5 if requested.

import h5py as h5
from netCDF4 import Dataset
import numpy as np
import pystare as ps
import json
from sortedcontainers import SortedDict, SortedList

import geodata as gd

try:
    from geodata.stopwatch import sw_timer
except ImportError:
    from stopwatch import sw_timer


def hex16(i):
    return "0x%016x"%i

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

class join_goes_and_m2(object):

    def __init__(self,goes_datapath,goes_filenames,m2_datapath,m2_file_name,verbose_progress=True):
        self.goes_datapath = goes_datapath
        self.goes_filenames = goes_filenames
        self.m2_datapath = m2_datapath
        self.m2_file_name = m2_file_name
        self.verbose_progress = verbose_progress
        self.ok = True

        sw_timer.stamp('join_goes_and_m2-start')
    
        self.goes_bandnames = {
            "BAND_03":"goes_b3"
            ,"BAND_04":"goes_b4"
            ,"BAND_05":"goes_b5"
        }
        self.goes_filenames_valid = []
        for i in self.goes_filenames:
            self.goes_band     = i.split('.')[4]    
            if self.goes_band in self.goes_bandnames.keys():
                self.goes_filenames_valid.append(i)
        if len(self.goes_filenames_valid) == 0:
            print("*ERROR* Join and save -- no valid GOES filenames. Returning.")
            self.ok = False # TODO Raise exception
            return

        print('goes_filenames:       ',self.goes_filenames)
        # print('goes_filenames_valid: ',goes_filenames_valid)
        print('m2_file_name          ',self.m2_file_name)
        sw_timer.stamp('join_goes_and_m2-end')
        return

    def join(self):
        "Join goes files with a particular m2_file."
        
        sw_timer.stamp('join_goes_and_m2-join-start')
    
        ###########################################################################
        ##### MERRA-2

        sw_timer.stamp('join_goes_and_m2-start-merra2-start')
    
        m2_dLat    = 0.5
        m2_dLon    = 5.0/8.0
        m2_dLatkm  = m2_dLat * gd.re_km/gd.deg_per_rad
        m2_dLonkm  = m2_dLon * gd.re_km/gd.deg_per_rad

        sw_timer.stamp('join_goes_and_m2-start-merra2-dataset-start')
        m2_ds  = Dataset(self.m2_datapath+self.m2_file_name)
        sw_timer.stamp('join_goes_and_m2-start-merra2-dataset-end')

        m2_lat,m2_lon = np.meshgrid(m2_ds['lat'],m2_ds['lon'])
        m2_lat     = m2_lat.flatten()
        m2_lon     = m2_lon.flatten()
        m2_idx_ij  = np.arange(m2_lat.size,dtype=np.int64)
        m2_resolution = int(gd.resolution(m2_dLonkm*2))
        m2_indices = ps.from_latlon(m2_lat,m2_lon,m2_resolution)
        m2_term = gd.spatial_terminator(m2_indices)
        m2_tid     = gd.merra2_stare_time(m2_ds)
    
        sw_timer.stamp('join_goes_and_m2-start-merra2-end')

    
        ###########################################################################
        ##### GOES

        sw_timer.stamp('join_goes_and_m2-start-goes-start')
    
        self.igoes = 0
        self.goes_band = self.goes_filenames_valid[self.igoes].split('.')[4]
        self.goes_bandname = self.goes_bandnames[self.goes_band]

        sw_timer.stamp('join_goes_and_m2-start-goes-start-dataset-start')
        self.goes_ds = Dataset(self.goes_datapath+self.goes_filenames_valid[self.igoes])
        sw_timer.stamp('join_goes_and_m2-start-goes-start-dataset-end')

        goes_tid = gd.goes10_img_stare_time(self.goes_ds)
    
        ##### MERRA-2 at the GOES time
        fine_match = ps.cmp_temporal(np.array(goes_tid,dtype=np.int64),m2_tid)
        # print('fine_match: ',fine_match)
        # print('gtid:       ',[hex16(i) for i in goes_tid])
        # print('m2_tid:     ',[hex16(i) for i in m2_tid])
        m2_ifm     = np.nonzero(fine_match)[0]
        # print('m2_ifm: ',m2_ifm)
        if m2_ifm.size == 1:
            m2_dataDayI     = m2_ds['TQI'][m2_ifm,:,:]
            m2_dataDayL     = m2_ds['TQL'][m2_ifm,:,:]
            m2_dataDayV     = m2_ds['TQV'][m2_ifm,:,:]
        else:
            m2_dataDayI     = np.mean(m2_ds['TQI'][m2_ifm,:,:],0)
            m2_dataDayL     = np.mean(m2_ds['TQL'][m2_ifm,:,:],0)
            m2_dataDayV     = np.mean(m2_ds['TQV'][m2_ifm,:,:],0)
            
        m2_dataDay      = m2_dataDayI + m2_dataDayL + m2_dataDayV
        # print('m2_dataDay.shape: ',m2_dataDay.shape)
        m2_data         = m2_dataDay[:,:].T
        # print('m2_data.shape:    ',m2_data.shape)
        m2_data_flat    = m2_data.flatten()
        # print('m2_data.flat.sh:  ',m2_data_flat.shape)
    
        g_lat = self.goes_ds['lat'][:,:].flatten()
        self.g_lat_size = g_lat.size
        # print('gds lat shape: ',self.goes_ds['lat'][:,:].shape)
        # print('gds lat fltsh: ',g_lat.shape)
        g_lon = self.goes_ds['lon'][:,:].flatten()
        g_idx_valid = np.where((g_lat>=-90.0) & (g_lat<=90.0))
        g_idx_invalid = np.where(((g_lat<-90.0) | (g_lat>90.0)))
        self.goes_indices = np.full(g_lat.shape,-1,dtype=np.int64)
        sw_timer.stamp('join_goes_and_m2-start-goes-from_latlon-start')
        self.goes_indices[g_idx_valid] = ps.from_latlon(g_lat[g_idx_valid],g_lon[g_idx_valid],int(gd.resolution(self.goes_ds['elemRes'][0])))
        self.goes_indices = self.goes_indices
        sw_timer.stamp('join_goes_and_m2-start-goes-from_latlon-end')
    
        ###########################################################################
        ##### Allocate MERRA-2 arrays co-aligned with GOES
        self.m2_src_coord_h5 = np.full(g_lat.shape,-1,dtype=np.int64)
        self.m2_tpw_h5       = np.full(g_lat.shape,-1,dtype=np.int64)
    
        sw_timer.stamp('join_goes_and_m2-start-join')
    
        #####
    
        join_resolution = m2_resolution
        join = SortedDict()
        
        ktr=0
        for k in range(len(g_idx_valid[0])):
            id = g_idx_valid[0][k]
            jk = gd.spatial_clear_to_resolution(gd.spatial_coerce_resolution(self.goes_indices[id],join_resolution))
            if jk not in join.keys():
                join[jk] = join_value()
            join[jk].add(self.goes_bandname,id)
            ktr = ktr + 1; 
            # if ktr > 10:
            #     break
            #     # exit();
        
        for k in range(len(m2_indices)):
            jk = gd.spatial_clear_to_resolution(m2_indices[k])
            if jk not in join.keys():
                join[jk] = join_value()
            join[jk].add('m2',k)
    
        ###########################################################################
        
        self.tpw_scale  = 0.001;
        self.tpw_offset = 0;
    
        ###########################################################################
        ##### JOIN
    
        # TODO Add metadata for traceability.
    
        jkeys=join.keys()
        ktr = 0; nktr = len(jkeys) # gd_idx_valid is a tuple with one element
        ktr_max = nktr
        elements_pushed = 0
        print('Push joined m2 data into the dataset n = ',nktr)
        for k in range(nktr):
            ktr = ktr + 1
            if self.verbose_progress:
                if int(100.0*ktr/ktr_max) % 5 == 0 or int(100*ktr/ktr_max) < 2:
                    print('join_goes_merra2: %2d%% complete, %d elements pushed.'%(int(100*ktr/ktr_max),elements_pushed),end='\r',flush=True)
            sid = jkeys[k]
            if join[sid].contains(self.goes_bandname):
                if join[sid].contains('m2'):
                    m2s = join[sid].get('m2')[0] # Grab the first one
                    self.m2_src_coord_h5[join[sid].get(self.goes_bandname)] = m2s
                    # self.m2_tpw_h5[join[sid].get(self.goes_bandname)]       = (m2_data_flat[m2s]-self.tpw_offset)/self.tpw_scale
                    avg = (np.mean(m2_data_flat[join[sid].get('m2')])-self.tpw_offset)/self.tpw_scale
                    self.m2_tpw_h5[join[sid].get(self.goes_bandname)]       = avg
                    elements_pushed = elements_pushed + len(join[sid].get(self.goes_bandname))
        print('join_goes_merra2: done, %d elements pushed.           '%(elements_pushed),flush=True)
        print('')
        sw_timer.stamp('join_goes_and_m2-join-end')    
        sw_timer.stamp('join_goes_and_m2-start-goes-end')
        return self
    

    def to_h5(self,workFileName):
        ###########################################################################
        ##### HDF5 SAVE DATASET

        sw_timer.stamp('join_goes_and_m2-to_h5-start')

        ##### HDF5 Data types for output
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
        image_description_dtype = np.dtype([
            ('nx',np.int)
            ,('ny',np.int)
        ])
        m2_description_dtype = np.dtype([
            ('nx',np.int)
            ,('ny',np.int)
            ,('self.tpw_offset',np.double)
            ,('self.tpw_scale',np.double)
        ])

        # self.m2_src_coord_h5
        # self.m2_tpw_h5
        # self.tpw_offset
        # self.tpw_scale
        # self.igoes
        # self.goes_bandname
        # self.goes_band
        # self.goes_filenames_valid
        # self.goes_datapath

        ##### HDF5 OUTPUT
        workFile = h5.File(workFileName,'w')
        image_ds = workFile.create_dataset('image',[self.goes_ds['data'].size],dtype=image_dtype)
        image_description_ds = workFile.create_dataset('image_description',[],dtype=image_description_dtype)
        m2_description_ds = workFile.create_dataset('merra2_description',[],dtype=m2_description_dtype)
        
        workFile['/image']['stare_spatial'] = self.goes_indices[:]
        workFile['/image']['stare_temporal'] = gd.goes10_img_stare_time(self.goes_ds)[0]
        workFile['/image']['goes_src_coord'] = np.arange(self.g_lat_size,dtype=np.int64)
        workFile['/image']['merra2_src_coord'] = self.m2_src_coord_h5.flatten()
        workFile['/image']['merra2_tpw']       = self.m2_tpw_h5.flatten()
    
        workFile['/image_description']['nx'] = self.goes_ds['data'].shape[2]
        workFile['/image_description']['ny'] = self.goes_ds['data'].shape[1]
    
        workFile['/merra2_description']['nx'] = 576
        workFile['/merra2_description']['ny'] = 361
        workFile['/merra2_description']['self.tpw_offset'] = self.tpw_offset
        workFile['/merra2_description']['self.tpw_scale']  = self.tpw_scale
    
        sw_timer.stamp('join_goes_and_m2-to_h5-goes-before-loop')
        while self.igoes < len(self.goes_filenames_valid):
            sw_timer.stamp('join_goes_and_m2-to_h5-goes-loop-start')
            print(self.igoes,' saving ',self.goes_bandname,' from file ',self.goes_filenames_valid[self.igoes])
            workFile['/image'][self.goes_bandname] = self.goes_ds['data'][0,:,:].flatten()
            self.goes_ds.close()
            self.igoes = self.igoes + 1
            # Assume remaining GOES bands have the same image sizes and locations.
            if self.igoes < len(self.goes_filenames_valid):
                self.goes_band     = self.goes_filenames_valid[self.igoes].split('.')[4]
                sw_timer.stamp('join_goes_and_m2-to_h5-goes-dataset-start')
                self.goes_ds       = Dataset(self.goes_datapath+self.goes_filenames_valid[self.igoes])
                sw_timer.stamp('join_goes_and_m2-to_h5-goes-dataset-end')
                self.goes_bandname = self.goes_bandnames[self.goes_band]
            sw_timer.stamp('join_goes_and_m2-to_h5-goes-loop-end')

        workFile.close()
        sw_timer.stamp('join_goes_and_m2-to_h5-end')
    
        return

def join_goes_and_m2_to_h5(goes_datapath,goes_filenames,m2_datapath,m2_file_name,workFileName,verbose_progress=True):
    return join_goes_and_m2(goes_datapath,goes_filenames,m2_datapath,m2_file_name,verbose_progress).join().to_h5(workFileName)
