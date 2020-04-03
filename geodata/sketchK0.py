
# Load GOES and MODIS, lexsort, look at nadir and wing footprints

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
from netCDF4 import Dataset

import modis_coarse_to_fine_geolocation.modis_5km_to_1km_geolocation as pascal_modis

###########################################################################

def with_hdf_get(h,var):
    sds = hdf.select(var)
    ret = sds.get()
    sds.endaccess()
    return ret

###########################################################################

def make_box(x,y,delta):
    return [x-delta,x+delta],[y-delta,y+delta]

def plot_box(ax,x_range,y_range,c=None):
    ax.plot([x_range[0],x_range[1],x_range[1],x_range[0],x_range[0]]
            ,[y_range[0],y_range[0],y_range[1],y_range[1],y_range[0]]
            ,c=c)
    return

def plot1(lon,lat,lons,lats,triang,c0='r',c1='b',transf=None,lw=1,ax=None):
    if(lon is not None and ax is not None):
        x=np.zeros([lon.size+1],dtype=np.double);x[:-1]=lon[:];x[-1]=lon[0]
        y=np.zeros([lat.size+1],dtype=np.double); y[:-1]=lat[:]; y[-1]=lat[0]
        ax.plot(x,y,True,transform=transf,c=c0)
    ax.triplot(triang,c1+'-',transform=transf,lw=lw,markersize=3)
    ax.scatter(lons,lats,s=10,c=c1,transform=transf)
    return

def plot_sivs(sivs,c0='r',c1='b',transf=None,lw=1,ax=None):
    lons0,lats0,intmat0 = ps.triangulate_indices(sivs)
    triang0 = tri.Triangulation(lons0,lats0,intmat0)
    plot1(None,None,lons0,lats0,triang0,c0=c0,c1=c1,transf=transf,ax=ax)
    return    

###########################################################################

def load_goes():

    # DEPENDENCIES
    goes_base    = "goes10."
    # goes_time    = "2005.349.003015"
    goes_time    = "2005.349.213015"

    ### GOES DATASET
    # Note GOES 10 (K) Bands 2-5 are 4-km, Band 1 (visible) is 1 km.
    goes_suffix = ".nc"
    goes_b5_dataPath = "/home/mrilee/data/GOES/"
    goes_b5_dataFile = goes_base+goes_time+".BAND_05"+goes_suffix
    goes_b5_fqFilename = goes_b5_dataPath+goes_b5_dataFile
    goes_b5_ds = Dataset(goes_b5_fqFilename)
    
    # goes_b3_dataPath = "/home/mrilee/data/"
    # goes_b3_dataFile = goes_base+goes_time+".BAND_03"+goes_suffix
    # goes_b3_fqFilename = goes_b3_dataPath+goes_b3_dataFile
    # goes_b3_ds = Dataset(goes_b3_fqFilename)

    # goes_b4_dataPath = "/home/mrilee/data/"
    # goes_b4_dataFile = goes_base+goes_time+".BAND_04"+goes_suffix
    # goes_b4_fqFilename = goes_b4_dataPath+goes_b4_dataFile
    # goes_b4_ds = Dataset(goes_b4_fqFilename)

    g5shape = goes_b5_ds['data'].shape
    print('g5shape = ',g5shape)
    
    g5size = goes_b5_ds['data'].size
    print('g5size = ',g5size)
    
    goes_b5_tid = gd.goes10_img_stare_time(goes_b5_ds)

    g5_dat = goes_b5_ds['data'][:,:].flatten()
    g5_lat = goes_b5_ds['lat'][:,:].flatten()
    g5_lon = goes_b5_ds['lon'][:,:].flatten()
    g5_idx_valid = np.where((g5_lat>=-90.0) & (g5_lat<=90.0))
    g5_idx_invalid = np.where(((g5_lat<-90.0) | (g5_lat>90.0)))
    g5_lon = g5_lon[g5_idx_valid]
    g5_lat = g5_lat[g5_idx_valid]
    g5_dat = g5_dat[g5_idx_valid]
    
    return g5_lon,g5_lat,g5_dat


###########################################################################

def load_modis():

    dataPath="/home/mrilee/data/MODIS/"
    
    # MOD03.A2005349.2120.061.2017187042837.hdf  MOD05_L2.A2005349.2120.061.2017294065852.hdf
    # MOD03.A2005349.2125.061.2017187042720.hdf  MOD05_L2.A2005349.2125.061.2017294065400.hdf
    
    modis_base   = "MOD05_L2."
    
    # 1 modis_item       = "A2005349.2120.061.2017294065852"
    # 1 modis_time_start = "2005-12-15T21:20:00"
    
    modis_item       = "A2005349.2125.061.2017294065400"
    modis_time_start = "2005-12-15T21:25:00"
    modis_geofilename = "MOD03.A2005349.2125.061.2017187042720.hdf"
    
    modis_suffix = ".hdf"
    modis_filename = modis_base+modis_item+modis_suffix
    
    fmt_suffix = ".h5"
    workFileName = "sketchG."+modis_base+modis_item+fmt_suffix
    
    key_across = 'Cell_Across_Swath_1km:mod05'
    key_along  = 'Cell_Along_Swath_1km:mod05'

    print('loading: ',dataPath+modis_filename)
    hdf        = SD(dataPath+modis_filename,SDC.READ)
    ds_wv_nir  = hdf.select('Water_Vapor_Near_Infrared')
    data       = ds_wv_nir.get()
    
    
    # MODIS_Swath_Type_GEO/Geolocation_Fields/
    # Latitude
    
    hdf_geo           = SD(dataPath+modis_geofilename,SDC.READ)
    print('hg info: ',hdf_geo.info())
    # for idx,sds in enumerate(hdf_geo.datasets().keys()):
    #     print(idx,sds)
    # hdf_geo_ds = hdf_geo.select['']
    
    # hdf_geo_swath     = hdf_geo.select('MODIS_Swath_Type_GEO')
    # hdf_geo_swath_gf  = hdf_geo_swath['Geolocation_Fields']
    hdf_geo_lat       = hdf_geo.select('Latitude').get()
    hdf_geo_lon       = hdf_geo.select('Longitude').get()
    print('hgl type  ',type(hdf_geo_lat))
    print('hgl shape ',hdf_geo_lat.shape,hdf_geo_lon.shape)
    print('hgl dtype ',hdf_geo_lat.dtype)
    # exit()
    
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

    mod_lat = hdf_geo_lat.flatten()
    mod_lon = hdf_geo_lon.flatten()
    mod_dat = data.flatten()

    return mod_lon,mod_lat,mod_dat # load_modis

def main():

    # GOES 10 (Band 5, 4km)
    # 0.036 deg
    g_lon,g_lat,g_dat = load_goes()
    g_lon_s,g_lat_s,g_dat_s,g_ilex = gd.lexsort_data(g_lon,g_lat,g_dat)
    
    # MODIS Water Vapor NIR 1km
    # 1.57e-4 radians -- 0.009 deg
    m_lon,m_lat,m_dat = load_modis()
    print('m_lon size: ',m_lon.size)
    m_str = np.zeros([m_lon.size],dtype=np.int64)
    for irow in range(2030):
        i0 = irow*1354
        i1 = (irow+1)*1354
        m_str[i0:i1]=ps.from_latlon(m_lat[i0:i1],m_lon[i0:i1],13) # 13 ~ 10km/(2**3)
    
    m_lon_s,m_lat_s,m_dat_s,m_ilex = gd.lexsort_data(m_lon,m_lat,m_dat)

    delta = 0.025 # degree
    mlon_mnmx = [np.amin(m_lon)-delta,np.amax(m_lon)+delta]
    mlat_mnmx = [np.amin(m_lat)-delta,np.amax(m_lat)+delta]

    idlon1 = np.where((mlon_mnmx[0] <= g_lon_s) & (g_lon_s <= mlon_mnmx[1]))
    g_lon1 = g_lon_s[idlon1]
    g_lat1 = g_lat_s[idlon1]
    g_dat1 = g_dat_s[idlon1]

    idlat2 = np.where((mlat_mnmx[0] <= g_lat1) & (g_lat1 <= mlat_mnmx[1]))
    g_lon2 = g_lon1[idlat2]
    g_lat2 = g_lat1[idlat2]
    g_dat2 = g_dat1[idlat2]

    g_lon3,g_lat3,g_dat3,g_ilex3 = gd.lexsort_data(g_lon2,g_lat2,g_dat2)

    viz = True
    viz_fig1_NadirVsWing = False
    
    if viz:
        proj   = ccrs.PlateCarree()
        transf = ccrs.Geodetic()
        nrows=2
        fig, axs = plt.subplots(nrows=nrows,subplot_kw={'projection':proj,'transform':transf})
        # plt.figure()
        # plt.subplot(projection=proj,transform=transf)
        # ax=axs[0]
        # ax.set_global()
        # ax.coastlines()
        
    delta = 0.05 # degree
    goes_nadir_hdelta=0.036/2.0 # 1/2 width deg for 4km pixel
    mod_nadir_hdelta=0.009/2.0   # 1/2 width deg for 1km pixel
    for i in range(g_lon3.size):
    # if True:
    #    i = 0
        lo = g_lon3[i]
        la = g_lat3[i]
        da = g_dat3[i]
        lo_range = [ lo-delta, lo+delta ]
        la_range = [ la-delta, la+delta ]
        lo_r,la_r,da_r = gd.subset_data_from_lonlatbox(m_lon,m_lat,m_dat,lo_range,la_range)

        if len(da_r) > 4:
            idlo_p=np.where(lo_r > lo)[0]
            idlo_m=np.where(lo_r < lo)[0]
            idlo_skew = (1.0*(len(idlo_p)-len(idlo_m)))/(1.0*(len(idlo_p)+len(idlo_m)))

            idla_p=np.where(la_r > la)[0]
            idla_m=np.where(la_r < la)[0]
            idla_skew = (1.0*(len(idla_p)-len(idla_m)))/(1.0*(len(idla_p)+len(idla_m)))
            
            if viz and abs(idlo_skew) < 0.5 and abs(idla_skew) < 0.5:
                print(i,' i,len(da_r),skew ',len(da_r),len(idlo_p),len(idlo_m),idlo_skew,idla_skew)
                
                ax=axs[0]
                # ax.set_global()
                ax.coastlines()
                
                xlim=[lo-2*delta,lo+2*delta]
                ylim=[la-2*delta,la+2*delta]
                ax.set_xlim(xlim[0],xlim[1])
                ax.set_ylim(ylim[0],ylim[1])
                
                plot_box(ax,mlon_mnmx,mlat_mnmx)
                plot_box(ax,lo_range,la_range)

                for j in range(len(da_r)):
                    xr,yr=make_box(lo_r[j],la_r[j],delta=mod_nadir_hdelta) # NADIR resolution for MODIS
                    plot_box(ax,xr,yr,c='k')

                ax.scatter([lo],[la],s=8,c='r',transform=transf)
                g_xr,g_yr = make_box(lo,la,goes_nadir_hdelta)
                plot_box(ax,g_xr,g_yr,c='r')

                if False: # Lexical sorting
                    ax.scatter(m_lon_s[0:1354],m_lat_s[0:1354],s=2.5,c='k',transform=transf)

                if False: # All
                    ax.scatter(m_lon_s,m_lat_s,s=2.5,c='k',transform=transf)

                if True:
                    for irow in range(2030-10,2030):
                        i0 = irow*1354
                        i1 = (irow+1)*1354
                        if True:
                            for j in range(i0,i1):
                                xr,yr=make_box(m_lon[j],m_lat[j],delta=mod_nadir_hdelta) # NADIR resolution for MODIS
                                plot_box(ax,xr,yr,c='k')
                        ax.scatter(m_lon[i0:i1],m_lat[i0:i1],s=2.5,c='k',transform=transf)
                        ind  = m_str[i0:i1]
                        inda = ps.adapt_resolution_to_proximity(ind)
                        plot_sivs(inda,c0='g',c1='g',transf=transf,ax=ax)   

                iax = 1
                ax=axs[iax]
                ax.coastlines()

                la0 =   18.125
                lo0 = -161.267

                xlim=[lo0-2*delta,lo0+2*delta]
                ylim=[la0-2*delta,la0+2*delta]
                print('iax,xlim,ylim: ',iax,xlim,ylim)
                ax.set_xlim(xlim[0],xlim[1])
                ax.set_ylim(ylim[0],ylim[1])

                if True:
                    for irow in range(2030-10,2030):
                        i0 = irow*1354
                        i1 = (irow+1)*1354
                        if True:
                            for j in range(i0,i1):
                                xr,yr=make_box(m_lon[j],m_lat[j],delta=mod_nadir_hdelta) # NADIR resolution for MODIS
                                plot_box(ax,xr,yr,c='k')
                        ax.scatter(m_lon[i0:i1],m_lat[i0:i1],s=2.5,c='k',transform=transf)
                        ind  = m_str[i0:i1]
                        inda = ps.adapt_resolution_to_proximity(ind)
                        plot_sivs(inda,c0='g',c1='g',transf=transf,ax=ax)
                
                    
                plt.show()
                return
        
    print('done')
    return

if __name__ == '__main__':
    main()
