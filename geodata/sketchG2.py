
from dask.distributed import Client

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

import yaml

from sortedcontainers import SortedDict, SortedList

from stopwatch import sw_timer

###########################################################################
#

def npi64(i):
    return np.array(i,dtype=np.int64)

def npf64(i):
    return np.array(i,dtype=np.double)

class sid_geometry(object):
    def __init__(self,sids=None):
        self.triangles = SortedDict()
        self.tri_triang    = None
        if sids is not None:
            self.add(sids)
        return

    def add(self,sids):
        for sid in sids:
            if sid not in self.triangles.keys():
                self.tri_triang = None
                self.triangles[sid] = ps.triangulate_indices(np.array([sid],dtype=np.int64)) ## LLI
        return

    def triang(self):
        if self.tri_triang is None:
            k=0
            n = len(self.triangles.keys())
            lats = np.zeros([3*n],dtype=np.double)
            lons = np.zeros([3*n],dtype=np.double)
            intmat = []
            for sid in self.triangles:
                lats[k:k+3]   = self.triangles[sid][0][:]
                lons[k:k+3]   = self.triangles[sid][1][:]
                intmat.append([i+k for i in self.triangles[sid][2][0]])
                k=k+3
            self.tri_triang = tri.Triangulation(lats,lons,intmat)
        return self.tri_triang

    def get_sids_np(self):
        return np.array(self.triangles.keys(),dtype=np.int64)

class data_entry(object):
    def __init__(self,sid,datum):
        self.sid = sid
        self.datum = datum
        return

    def as_tuple(self):
        return (self.sid,self.datum)

class catalog_entry(object):
    def __init__(self,sid):
        self.data = {}
        self.sid  = sid
        self.geometry = sid_geometry([sid])
        return

    def add(self,key,datum):
        if key not in self.data.keys():
            self.data[key] = []
        self.data[key].append(datum) # datum is a data_entry
        return

class catalog(object):
    def __init__(self,resolution=None,sids=None):
        self.resolution = resolution
        self.result_size_limit = 4096
        self.sdict      = SortedDict()
        if sids is not None:
            for s in sids:
                self.open_entry(s)
        return

    def add(self,key,sid,datum,resolution=None):
        if resolution is not None:
            sidtest = gd.spatial_clear_to_resolution(gd.spatial_coerce_resolution(sid,resolution))
        elif self.resolution is not None:
            sidtest = gd.spatial_clear_to_resolution(gd.spatial_coerce_resolution(sid,self.resolution))
        else:
            sidtest = sid
        if sidtest not in self.sdict.keys():
            self.sdict[sidtest] = catalog_entry(sidtest) # construct with relevant resolution
        self.sdict[sidtest].add(key,data_entry(sid,datum))
        return

    def open_entry(self,sid):
        "Open a catalog entry, if it's not there. Expand sid, if needed."
        sidl=[sid]
        if self.resolution is not None:
            sidl = ps.expand_intervals(sidl,self.resolution,self.result_size_limit)
        for s in sidl:
            if s not in self.sdict.keys():
                self.sdict[s] = catalog_entry(s) # construct with appropriate resolution
        return

    def add_to_entry(self,key,sid,datum):
        "Add data to entry if it's there."
        if self.resolution is not None:
            sid_test = gd.spatial_clear_to_resolution(gd.spatial_coerce_resolution(sid,self.resolution))
        else:
            sid_test = sid
        # print('testing ',hex(sid_test), hex(sid))
        entry_open = sid_test in self.sdict.keys()
        if entry_open:
            # print(key,' adding ',data,' to ',hex(sid))
            self.add(key,sid,datum)
        return entry_open

    def get_all_data(self,key,interpolate=False):
        ret = []
        for s in self.sdict.keys():
            try:
                if len(self.sdict[s].data[key]) > 0:
                    ret = ret + self.sdict[s].data[key]
            except KeyError:
                continue
        return ret

    def get_data(self,key,sid):
        return self.sdict[sid].data[key]
            

###########################################################################
# Helper functions

def with_hdf_get(h,var):
    sds = hdf.select(var)
    ret = sds.get()
    sds.endaccess()
    return ret

def slam(client,action,data,partition_factor=1.5):
    np = sum(client.nthreads().values())
    print('slam: np = %i'%np)
    shard_bounds = [int(i*len(data)/(1.0*partition_factor*np)) for i in range(int(partition_factor*np))] 
    if shard_bounds[-1] != len(data):
        shard_bounds = shard_bounds + [-1]
    data_shards = [data[shard_bounds[i]:shard_bounds[i+1]] for i in range(len(shard_bounds)-1)]
    print('ds len:        ',len(data_shards))
    print('ds item len:   ',len(data_shards[0]))
    print('ds type:       ',type(data_shards[0]))
    print('ds dtype:      ',data_shards[0].dtype)
    big_future = client.scatter(data_shards)
    results    = client.map(action,big_future)
    return results
    

def main():
    ###########################################################################
    # Data source
    dataPath="/home/mrilee/data/"
    
    ###########################################################################
    # MODIS

    modis_base   = "MOD05_L2."
    
    # modis_item   = "A2005349.2120.061.2017294065852"
    # modis_time_start = "2005-12-15T21:20:00"
    
    modis_item       = "A2005349.2125.061.2017294065400"
    modis_time_start = "2005-12-15T21:25:00"
    
    modis_suffix = ".hdf"
    modis_filename = modis_base+modis_item+modis_suffix
    
    fmt_suffix = ".h5"
    workFileName = "sketchG."+modis_base+modis_item+fmt_suffix
    print('loading ',workFileName)
    workFile = h5.File(workFileName,'r')
    sids = workFile['/image']['stare_spatial']
    lat  = workFile['/image']['Latitude']
    lon  = workFile['/image']['Longitude']
    data = workFile['/image']['Water_Vapor_Near_Infrared']
    workFile.close()

    modis_min = np.amin(data)
    modis_max = np.amax(data)
    sids = sids-1

    ###########################################################################
    # MODIS HDF

    hdf        = SD(dataPath+modis_filename,SDC.READ)
    # ds_wv_nir  = hdf.select('Water_Vapor_Near_Infrared')

    modis_hdf_resolution = 7
    ntri_max   = 1000
    # hull = ps.to_hull_range_from_latlon(gring_lat[gring_seq],gring_lon[gring_seq],resolution,ntri_max)
    modis_hdf_hull = gd.modis_cover_from_gring(hdf,modis_hdf_resolution,ntri_max)
    hdf.end()
    
    modis_hdf_lons,modis_hdf_lats,modis_hdf_intmat = ps.triangulate_indices(modis_hdf_hull)
    modis_hdf_triang = tri.Triangulation(modis_hdf_lons,modis_hdf_lats,modis_hdf_intmat)


    ###########################################################################
    # GOES
    
    goes_file='sketch9.2005.349.213015.h5'
    workFileName = goes_file
    workFile = h5.File(workFileName,'r')
    goes_sids = workFile['/image']['stare_spatial']
    goes_data = workFile['/image']['goes_b3']
    workFile.close()
    print('goes mnmx: ',np.amin(goes_data),np.amax(goes_data))
    goes_min = np.amin(goes_data)
    goes_max = np.amax(goes_data)
    goes_sids = goes_sids-1

    goes_sids_005 = np.unique(gd.spatial_clear_to_resolution((goes_sids[np.where(goes_sids>0)] & ~31)+5))

    goes_sids_005_geom   = sid_geometry(goes_sids_005)
    goes_sids_005_triang = goes_sids_005_geom.triang()

    ###########################################################################
    # Plotting an ROI
    cover_rads =[2.0,0,0, 0.125,0,0]
    circle_color=[ 'Lime' ,'lightgrey' ,'White' ,'navajowhite' ,'khaki' ,'White' ]
    ###########################################################################
    # HI 28.5N 177W
    cover_resolution = 11
    cover_lat =   19.5-0.375
    cover_lon = -155.5+0.375

    ###########################################################################
    # Plotting

    nrows = 1
    ncols = 1
    # proj   = ccrs.PlateCarree()
    proj   = ccrs.PlateCarree(central_longitude=-160.0)
    # proj   = ccrs.Mollweide()
    # proj   = ccrs.Mollweide(central_longitude=-160.0)
    transf = ccrs.Geodetic()

    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,subplot_kw={'projection': proj})

    subplot_title = ['GOES and MODIS Granules']
    ax = axs

    distribute_to_nodes = False

    if True:
        iter = 0

        ax.set_xlim(-160-45,160+45)

        if subplot_title[iter] is not None:
            ax.set_title(subplot_title[iter])
        if False:
            ax.set_global()
        if True:
            ax.coastlines()

        if distribute_to_nodes:
            vmin=0; vmax=16
            # colors=np.arange(goes_sids_005_triang.x.shape[0]/3,dtype=np.int64)%(vmax-vmin)
            colors=np.random.uniform(vmin,vmax,size=int(goes_sids_005_triang.x.shape[0]/3)).astype(np.int64)
            ax.tripcolor(goes_sids_005_triang
                         ,facecolors=colors
                         ,edgecolors='k',lw=0
                         ,shading='flat'
                         ,vmin=vmin,vmax=vmax,cmap='rainbow',alpha=0.65
                         ,transform=transf)

        ax.triplot(goes_sids_005_triang ,'r-',transform=transf,lw=1.0,markersize=3,alpha=0.5)
        ax.triplot(modis_hdf_triang     ,'b-',transform=transf,lw=1.0,markersize=3,alpha=0.5)

        if True:
            phi=np.linspace(0,2*np.pi,64)
            cover_rad = cover_rads[iter]
            rad=cover_rad
            # rad=0.125
            ax.plot(cover_lon+rad*np.cos(phi),cover_lat+rad*np.sin(phi),transform=transf,color=circle_color[iter],lw=2)

        if iter == 0:
            x0 = 0.05
            y0 = 0.025; dy = 0.025
            plt.figtext(x0,y0+0*dy
                        ,"MODIS (blue): "+"sketchG."+modis_base+modis_item+fmt_suffix+', cover from GRING metadata, max resolution = 7'
                        ,fontsize=10)
            k=0;
            while goes_sids[k]<0:
                k=k+1
            plt.figtext(x0,y0+1*dy
                        ,"GOES (red):  "+goes_file+', Northern Hemisphere, resolution = 5'
                        ,fontsize=10)
            plt.figtext(x0,y0+2*dy
                        ,"Region of Interest near Hawaii, ROI (%s): radius = %0.2f degrees, center = 0x%016x"%(circle_color[iter],cover_rads[0],ps.from_latlon(npf64([cover_lat]),npf64([cover_lon]),cover_resolution)[0])
                        ,fontsize=10)

            if distribute_to_nodes:
                plt.figtext(x0,y0+3*dy
                            ,"Color tiles show distribution across 16 nodes."
                            ,fontsize=10)

    plt.show()


if __name__ == "__main__":

    main()
