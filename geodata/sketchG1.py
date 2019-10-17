
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
            sidl = ps.expand_intervals(sidl,self.resolution)
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

    def get_all_data(self,key):
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

    # hdf        = SD(dataPath+modis_filename,SDC.READ)
    # ds_wv_nir  = hdf.select('Water_Vapor_Near_Infrared')
    
    fmt_suffix = ".h5"
    workFileName = "sketchG."+modis_base+modis_item+fmt_suffix
    print('loading ',workFileName)
    workFile = h5.File(workFileName,'r')
    sids = workFile['/image']['stare_spatial']
    lat  = workFile['/image']['Latitude']
    lon  = workFile['/image']['Longitude']
    data = workFile['/image']['Water_Vapor_Near_Infrared']
    workFile.close()

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

    ###########################################################################
    # HI 28.5N 177W
    cover = ps.to_circular_cover(   28.5
                                 ,-177.0
                                 ,   2.0
                                 ,   7)
    cover_cat = catalog(resolution=7,sids=cover)
    # for k in list(cover_cat.sdict.keys()):
    #     print('cc: ',hex(k))

    ###########################################################################
    #
    gm_catalog = catalog(resolution=7)
    k=0
    for i in range(10):
        while(goes_sids[k]<0):
            k=k+1
        print('adding: ','0x%016x'%goes_sids[k],k)
        gm_catalog.add('goes',goes_sids[k],goes_data[k])
        k=k+1

    for i in range(10):
        print('adding: ','0x%016x'%sids[i])
        gm_catalog.add('modis',sids[i],data[i])

    k = 0
    # for i in range(10):
    for k in range(len(goes_sids)):
        # while(goes_sids[k]<0):
        #    k=k+1
        if goes_sids[k] > 0:
            cover_cat.add_to_entry('goes',goes_sids[k],goes_data[k])
        # k=k+1

    # print(yaml.dump(gm_catalog))
    # exit()

    ###########################################################################
    # Plotting

    print('plotting')
    proj   = ccrs.PlateCarree()
    # proj   = ccrs.Mollweide()
    # proj   = ccrs.Mollweide(central_longitude=-160.0)
    transf = ccrs.Geodetic()
    
    sw_timer.stamp()
    plt.figure()
    ax = plt.axes(projection=proj)
    ax.set_title('MODIS STARE Test')
    ax.set_global()
    ax.coastlines()

    if False:
        # k = gm_catalog.sdict.keys()[0]
        # for k in gm_catalog.sdict.keys():
        for i in range(0,3):
            k = gm_catalog.sdict.peekitem(i)[0]
            triang = gm_catalog.sdict[k].geometry.triang()
            plt.triplot(triang,'r-',transform=transf,lw=1,markersize=3)

    if True:
        lli = ps.triangulate_indices(cover)
        plt.triplot(tri.Triangulation(lli[0],lli[1],lli[2])
                    ,'g-',transform=transf,lw=1,markersize=3)

    if True:
        cc_data = cover_cat.get_all_data('goes')
        # print('a100: ',cc_data)
        for cd in cc_data:
            lli    = ps.triangulate_indices([cd.sid])
            triang = tri.Triangulation(lli[0],lli[1],lli[2])
            cd_plt = np.array([cd.datum])
            plt.triplot(triang,'r-',transform=transf,lw=1,markersize=3,alpha=0.5)
            plt.tripcolor(triang,facecolors=cd_plt,vmin=goes_min,vmax=goes_max,cmap='rainbow')

        for i in range(0,10):
            k = cover_cat.sdict.peekitem(i)[0]
            triang = cover_cat.sdict[k].geometry.triang()
            plt.triplot(triang,'b-',transform=transf,lw=1,markersize=3,alpha=0.5)
            

    if False:
        sw_timer.stamp('triangulating')
        print('triangulating')
        client = Client()
        for lli_ in slam(client,ps.triangulate_indices,sids):
            sw_timer.stamp('slam iteration')
            print('lli_ type: ',type(lli_))
            lli = lli_.result()
            sw_timer.stamp('slam result')
            print('lli type:  ',type(lli))
            triang = tri.Triangulation(lli[0],lli[1],lli[2])
            sw_timer.stamp('slam triang')
            plt.triplot(triang,'r-',transform=transf,lw=1.5,markersize=3,alpha=0.5)
            sw_timer.stamp('slam triplot')

    sw_timer.stamp('plt show')
    # lons,lats,intmat=ps.triangulate_indices(sids)
    # triang = tri.Triangulation(lons,lats,intmat)
    # plt.triplot(triang,'r-',transform=transf,lw=1.5,markersize=3)

    plt.show()

    client.close()

    print(sw_timer.report_all())

if __name__ == "__main__":

    main()

