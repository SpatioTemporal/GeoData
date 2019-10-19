
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

    modis_min = np.amin(data)
    modis_max = np.amax(data)
    sids = sids-1

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


    ###########################################################################
    # Plotting

    nrows = 2
    ncols = 3
    proj   = ccrs.PlateCarree()
    # proj   = ccrs.Mollweide()
    # proj   = ccrs.Mollweide(central_longitude=-160.0)
    transf = ccrs.Geodetic()

# https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    # plt.figure()
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,subplot_kw={'projection': proj})

    goes_line          = [False,False,False]
    modis_line         = [False,False,False]
    cover_plot         = [True, True, True ]
    goes_plot_1        = [True, False,True ]
    goes_plot_1_points = [True, False,True ]
    modis_plot_1       = [False,True, True ]
    plt_show_1         = [False,False,True ]

    goes_line           = [False,False,False  ,True  ,False ,True   ]
    modis_line          = [False,False,False  ,False ,True  ,True   ]
    cover_plot          = [False,False,False  ,False ,False ,False  ]
    goes_plot_1         = [True, False,True   ,True  ,False ,True   ]
    goes_plot_1_points  = [False,False,False  ,True  ,False ,True   ]
    modis_plot_1        = [False,True, True   ,False ,True  ,True   ]
    modis_plot_1_points = [False,False,False  ,False ,True  ,True   ] 
    plt_show_1          = [False,False,False  ,False ,False ,True   ]
    
    irow = [0,0,0,1,1,1]
    icol = [0,1,2,0,1,2]

    recalculate=[True,False,False,True,False,False]
    cover_rads =[2.0,0,0, 0.125,0,0]

    circle_color=[ 'White' ,'Grey' ,'White' ,'White' ,'White' ,'White' ]

    subplot_title = [
        "ROI+GOES"
        ,"ROI+MODIS"
        ,"ROI+GOES+MODIS"
        ,None
        ,None
        ,None
    ]
    
    for iter in range(6):

        ###########################################################################
        if recalculate[iter]:
            print('recalculating iter = ',iter)

            ###########################################################################
            # HI 28.5N 177W
            cover_resolution = 11
            cover_lat =   19.5-0.375
            cover_lon = -155.5+0.375
            cover_rad = cover_rads[iter]
            
            cover = ps.to_circular_cover(
                cover_lat
                ,cover_lon
                ,cover_rad
                ,cover_resolution
                ,range_size_limit=2000)
        
            cover_cat = catalog(resolution=11,sids=cover)
            cover_sids_min = np.amin(cover)
            cover_sids_max = np.amax(cover) # Need to convert to terminator
            cover_sids_max = gd.spatial_terminator(cover_sids_max)
            # for k in list(cover_cat.sdict.keys()):
            #     print('cc: ',hex(k))
        
            ###########################################################################
            #
            gm_catalog = catalog(resolution=7)
            k=0
            for i in range(10):
                while(goes_sids[k]<0):
                    k=k+1
                # print('adding: ','0x%016x'%goes_sids[k],k)
                gm_catalog.add('goes',goes_sids[k],goes_data[k])
                k=k+1
        
            for i in range(10):
                # print('adding: ','0x%016x'%sids[i])
                gm_catalog.add('modis',sids[i],data[i])
        
            k = 0
            # for i in range(10):
            idx = np.arange(goes_sids.size)[np.where( (goes_sids > cover_sids_min) & (goes_sids < cover_sids_max))]
            for k in range(len(idx)):
                # while(goes_sids[k]<0):
                #    k=k+1
                if goes_sids[idx[k]] > 0:
                    cover_cat.add_to_entry('goes',goes_sids[idx[k]],goes_data[idx[k]])
                # k=k+1
        
            idx = np.arange(sids.size)[np.where( (sids > cover_sids_min) & (sids < cover_sids_max))]
            for k in range(len(idx)):
                if sids[idx[k]] > 0:
                    cover_cat.add_to_entry('modis',sids[idx[k]],data[idx[k]])
        
        
            # print(yaml.dump(gm_catalog))
            # exit()
            #
            ###########################################################################

        print('plotting iter ',iter)
        
        ax = axs[irow[iter],icol[iter]]
        
        if subplot_title[iter] is not None:
            ax.set_title(subplot_title[iter])
        if False:
            ax.set_global()
        if True:
            ax.coastlines()
    
        if False:
            # k = gm_catalog.sdict.keys()[0]
            # for k in gm_catalog.sdict.keys():
            for i in range(0,3):
                k = gm_catalog.sdict.peekitem(i)[0]
                triang = gm_catalog.sdict[k].geometry.triang()
                ax.triplot(triang,'r-',transform=transf,lw=1,markersize=3)
    
        if False:
            lli = ps.triangulate_indices(cover)
            ax.triplot(tri.Triangulation(lli[0],lli[1],lli[2])
                        ,'g-',transform=transf,lw=1,markersize=3)
    
        if True:
            if goes_plot_1[iter]:
                cc_data = cover_cat.get_all_data('goes')
                csids,sdat = zip(*[cd.as_tuple() for cd in cc_data])
                glat,glon = ps.to_latlon(csids)

                csids_at_res = list(map(gd.spatial_clear_to_resolution,csids))
                cc_data_accum = dict()
                for cs in csids_at_res:
                    cc_data_accum[cs] = []
                for ics in range(len(csids_at_res)):
                    cc_data_accum[csids_at_res[ics]].append(sdat[ics])
                for cs in cc_data_accum.keys():
                    if len(cc_data_accum[cs]) > 1:
                        cc_data_accum[cs] = [sum(cc_data_accum[cs])/(1.0*len(cc_data_accum[cs]))]
                tmp_values = np.array(list(cc_data_accum.values()))
                vmin = np.amin(tmp_values)
                vmax = np.amax(np.array(tmp_values))

                # print('a100: ',cc_data)
                # print('cc_data       type: ',type(cc_data))
                # print('cc_data[0]    type: ',type(cc_data[0]))
                
                for cs in cc_data_accum.keys():
                    # print('item: ',hex(cs),cc_data_accum[cs])
                    lli    = ps.triangulate_indices([cs])
                    triang = tri.Triangulation(lli[0],lli[1],lli[2])
                    cd_plt = np.array(cc_data_accum[cs])
                    # print('cd_plt type ',type(cd_plt))
                    # print('cd_plt shape ',cd_plt.shape)
                    # print('cd_plt type ',type(cd_plt[0]))
                    if goes_line[iter]:
                        ax.triplot(triang,'r-',transform=transf,lw=1.5,markersize=3,alpha=0.5)
                    # ax.tripcolor(triang,facecolors=cd_plt,vmin=goes_min,vmax=goes_max,cmap='Reds',alpha=0.4)
                    ax.tripcolor(triang
                                 ,facecolors=cd_plt
                                 ,edgecolors='k',lw=0
                                 ,shading='flat'
                                 ,vmin=vmin,vmax=vmax,cmap='Reds',alpha=0.45)
    
                # for cd in cc_data:
                #     lli    = ps.triangulate_indices([cd.sid])
                #     triang = tri.Triangulation(lli[0],lli[1],lli[2])
                #     cd_plt = np.array([cd.datum])
                #     if goes_line[iter]:
                #         ax.triplot(triang,'r-',transform=transf,lw=3,markersize=3,alpha=0.5)
                #     ax.tripcolor(triang,facecolors=cd_plt,vmin=goes_min,vmax=goes_max,cmap='Reds',alpha=0.4)
    
            if modis_plot_1[iter]:
                cc_data_m = cover_cat.get_all_data('modis')
                csids,sdat = zip(*[cd.as_tuple() for cd in cc_data_m])
                mlat,mlon = ps.to_latlon(csids)

                cc_data_m_accum,vmin,vmax = gd.simple_collect(csids,sdat)

                for cs in cc_data_m_accum.keys():
                    lli    = ps.triangulate_indices([cs])
                    triang = tri.Triangulation(lli[0],lli[1],lli[2])
                    cd_plt = np.array(cc_data_m_accum[cs])
                    # print('lli[0] len ',len(lli[0]))
                    # print('cd_plt len ', len(cd_plt))
                    # print('cd_plt type ',type(cd_plt))
                    # print('cd_plt shape ',cd_plt.shape)
                    # print('cd_plt type ',type(cd_plt[0]))
                    if modis_line[iter]:
                        ax.triplot(triang,'b-',transform=transf,lw=1.5,markersize=3,alpha=0.5)
                    # ax.tripcolor(triang,facecolors=cd_plt,vmin=goes_min,vmax=goes_max,cmap='Blues',alpha=0.4)
                    ax.tripcolor(triang
                                 ,facecolors=cd_plt
                                 ,edgecolors='k',lw=0
                                 ,shading='flat'
                                 ,vmin=vmin,vmax=vmax,cmap='Blues',alpha=0.45)

                # for cd in cc_data_m:
                #     lli    = ps.triangulate_indices([cd.sid])
                #     triang = tri.Triangulation(lli[0],lli[1],lli[2])
                #     cd_plt = np.array([cd.datum])
                #     if modis_line[iter]:
                #         ax.triplot(triang,'b-',transform=transf,lw=1,markersize=3,alpha=0.5)
                #     ax.tripcolor(triang,facecolors=cd_plt,vmin=modis_min,vmax=modis_max,cmap='Blues',alpha=0.4)
                if modis_plot_1_points[iter]:
                    ax.scatter(mlon,mlat,s=8,c='cyan')
                    # ax.scatter(mlon,mlat,s=8,c='darkcyan')

            if goes_plot_1[iter]:
                if goes_plot_1_points[iter]:
                    ax.scatter(glon,glat,s=8,c='black')
    
            if False:
                for i in range(0,10):
                    k = cover_cat.sdict.peekitem(i)[0]
                    triang = cover_cat.sdict[k].geometry.triang()
                    ax.triplot(triang,'b-',transform=transf,lw=1,markersize=3,alpha=0.5)

            if cover_plot[iter]:
                # lli = ps.triangulate_indices(ps.expand_intervals(cover,9,result_size_limit=2048))
                lli = ps.triangulate_indices(cover)
                ax.triplot(tri.Triangulation(lli[0],lli[1],lli[2])
                           ,'g-',transform=transf,lw=1,markersize=3)

            if True:
                phi=np.linspace(0,2*np.pi,64)
                # rad=cover_rad
                rad=0.125
                ax.plot(cover_lon+rad*np.cos(phi),cover_lat+rad*np.sin(phi),transform=transf,color=circle_color[iter])

            if plt_show_1[iter]:
                plt.show()
                
###########################################################################
#
#    if False:
#        sw_timer.stamp('triangulating')
#        print('triangulating')
#        client = Client()
#        for lli_ in slam(client,ps.triangulate_indices,sids):
#            sw_timer.stamp('slam iteration')
#            print('lli_ type: ',type(lli_))
#            lli = lli_.result()
#            sw_timer.stamp('slam result')
#            print('lli type:  ',type(lli))
#            triang = tri.Triangulation(lli[0],lli[1],lli[2])
#            sw_timer.stamp('slam triang')
#            plt.triplot(triang,'r-',transform=transf,lw=1.5,markersize=3,alpha=0.5)
#            sw_timer.stamp('slam triplot')
#
#    sw_timer.stamp('plt show')
#    # lons,lats,intmat=ps.triangulate_indices(sids)
#    # triang = tri.Triangulation(lons,lats,intmat)
#    # plt.triplot(triang,'r-',transform=transf,lw=1.5,markersize=3)
#
#    plt.show()

    client.close()

    print(sw_timer.report_all())

if __name__ == "__main__":

    main()

