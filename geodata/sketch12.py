
### Cross comparison m2 & b5.

import geodata as gd
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

import pystare as ps

from netCDF4 import Dataset

### MERRA2 
m2_dataPath = "/home/mrilee/data/"
m2_dataFile = "MERRA2_300.tavg1_2d_slv_Nx.20051215.nc4"
m2_fqFilename = m2_dataPath+m2_dataFile
m2_ds = Dataset(m2_fqFilename)
print('keys: ',m2_ds.variables.keys())
m2_dLat = 0.5
m2_dLon = 5.0/8.0
m2_dLatkm = m2_dLat * gd.re_km/gd.deg_per_rad
m2_dLonkm = m2_dLon * gd.re_km/gd.deg_per_rad
m2_lat0=0
m2_lat1=361+1
m2_lon0=0
m2_lon1=576+1
m2_tim0=0
m2_tim1=23+1
m2_lat     = m2_ds['lat'][m2_lat0:m2_lat1]
m2_lon     = m2_ds['lon'][m2_lon0:m2_lon1]
m2_tim     = m2_ds['time'][m2_tim0:m2_tim1]
m2_dataDayI = m2_ds['TQI'][m2_tim0:m2_tim1,m2_lat0:m2_lat1,m2_lon0:m2_lon1]
m2_dataDayL = m2_ds['TQL'][m2_tim0:m2_tim1,m2_lat0:m2_lat1,m2_lon0:m2_lon1]
m2_dataDayV = m2_ds['TQV'][m2_tim0:m2_tim1,m2_lat0:m2_lat1,m2_lon0:m2_lon1]
m2_dataDay = m2_dataDayI + m2_dataDayL + m2_dataDayV
m2_data    = m2_dataDay[0,:,:].T
m2_latg,m2_long = np.meshgrid(m2_lat,m2_lon)
m2_latg_flat = m2_latg.flatten()
m2_long_flat = m2_long.flatten()
m2_data_flat = m2_data.flatten()
print([type(i) for i in [m2_latg_flat,m2_long_flat,gd.resolution(m2_dLonkm),m2_dLonkm]])
m2_indices = ps.from_latlon(m2_latg_flat,m2_long_flat,int(gd.resolution(m2_dLonkm)))

print('m2 lat.shape:      ',m2_lat.shape)
print('m2 lon.shape:      ',m2_lon.shape)
print('m2 tim.shape:      ',m2_tim.shape)
print('m2 latg.shape:     ',m2_latg.shape)
print('m2 long.shape:     ',m2_long.shape)
print('m2 dataDay.shape:  ',m2_dataDay.shape)
print('m2 data.shape:     ',m2_data.shape)
print('m2 data flat shape:',m2_data_flat.shape)
print('m2 resolution:     ',m2_dLonkm,gd.resolution(m2_dLonkm))

# Limit size for testing
cropped = True
if cropped:
    # 1 box
    # crop_lat=(  30.0,  31.0)
    # crop_lon=(-150.0,-149.0)

    # crop_lat=(  30.0,  32.0)
    # crop_lon=(-164.0,-162.0)

    # crop_lat=(  25.0,  30.0) # oops
    # crop_lat=(  25.0,  35.0)
    # crop_lon=(-160.0,-150.0)

    crop_lat=(  27.0,  39.0)
    crop_lon=(-161.5,-159.5)

    # Good
    # crop_lat=(  27.5,  32.5)
    # crop_lon=(-162.5,-157.5)

    m2_crop_idx = np.where(\
                           (m2_latg_flat > crop_lat[0]) & (m2_latg_flat < crop_lat[1]) & \
                           (m2_long_flat > crop_lon[0]) & (m2_long_flat < crop_lon[1]) )
    m2_latg_flat = m2_latg_flat[m2_crop_idx]
    m2_long_flat = m2_long_flat[m2_crop_idx]
    m2_data_flat = m2_data_flat[m2_crop_idx]
    m2_indices   = m2_indices[m2_crop_idx]
    print('m2 cropped length: ', m2_data_flat.size)

ar_threshold_kgom2 = 2.0/gd.precipitable_water_cm_per_kgom2
ar_threshold_kgom2 = 0.0
print('ar_threshold_kgom2: ',ar_threshold_kgom2)

m2_ar_idx = np.where(m2_data_flat >= ar_threshold_kgom2)
print('m2_ar size idx: ',len(m2_ar_idx[0]))
if len(m2_ar_idx[0]) is  0:
    print('no m2_ar found!'); exit()

# print('m2 idx:      ',m2_ar_idx)
m2_ar_data    = m2_data_flat[m2_ar_idx]
m2_ar_indices = m2_indices[m2_ar_idx]
m2_ar_lat     = m2_latg_flat[m2_ar_idx]
m2_ar_lon     = m2_long_flat[m2_ar_idx]
print('m2_ar_lat mnmx: ',np.amin(m2_ar_lat),np.amax(m2_ar_lat))
print('m2_ar_lon mnmx: ',np.amin(m2_ar_lon),np.amax(m2_ar_lon))

# exit()

if False:
    nbins = 10
    plt.figure()
    n,bins,patches = plt.hist(m2_data.flatten(),nbins,facecolor='blue',alpha=0.5)
    plt.show()
    exit()

### GOES BAND_05
goes_b5_dataPath = "/home/mrilee/data/"
goes_b5_dataFile = "goes10.2005.349.003015.BAND_05.nc"
goes_b5_fqFilename = goes_b5_dataPath+goes_b5_dataFile
goes_b5_ds = Dataset(goes_b5_fqFilename)
goes_b5_lat0=0
goes_b5_lat1=1355+1
goes_b5_lon0=0
goes_b5_lon1=3311+1
goes_b5_lat  = goes_b5_ds['lat'][goes_b5_lat0:goes_b5_lat1,goes_b5_lon0:goes_b5_lon1]
goes_b5_lon  = goes_b5_ds['lon'][goes_b5_lat0:goes_b5_lat1,goes_b5_lon0:goes_b5_lon1]
goes_b5_data = goes_b5_ds['data'][0,goes_b5_lat0:goes_b5_lat1,goes_b5_lon0:goes_b5_lon1]
# print('goes_b5_elemRes = ',goes_b5_ds['elemRes'])
# print('goes_b5_elemRes = ',goes_b5_ds['elemRes'].long_name)
# print('goes_b5_elemRes = ',goes_b5_ds['elemRes'][0])
# print('goes_b5_elemRes = ',goes_b5_ds['elemRes'].units)
# print('goes_b5_elemRes = ',goes_b5_ds['lineRes'])
# print('goes_b5_elemRes = ',goes_b5_ds['lineRes'][0])
# exit()
goes_b5_elemRes = goes_b5_ds['elemRes'][0]*0.5
goes_b5_lat_flat = goes_b5_lat.flatten()
goes_b5_lon_flat = goes_b5_lon.flatten()
goes_b5_data_flat = goes_b5_data.flatten()
goes_b5_indices = ps.from_latlon(goes_b5_lat_flat,goes_b5_lon_flat,int(gd.resolution(goes_b5_elemRes)))
print('goes_b5_indices size: ',goes_b5_indices.size)
print('goes_b5_indices type: ',type(goes_b5_indices.size))
print('goes_b5 resolution:   ',goes_b5_elemRes,gd.resolution(goes_b5_elemRes))
# exit()

if False:
    plt.figure()
    plt.imshow(goes_b5_data)
    plt.show()

exit()

if cropped:
    goes_b5_crop_idx = np.where(\
                           (goes_b5_lat_flat > crop_lat[0]) & (goes_b5_lat_flat < crop_lat[1]) & \
                           (goes_b5_lon_flat > crop_lon[0]) & (goes_b5_lon_flat < crop_lon[1]) )
    goes_b5_lat_flat = goes_b5_lat_flat[goes_b5_crop_idx]
    goes_b5_lon_flat = goes_b5_lon_flat[goes_b5_crop_idx]
    goes_b5_indices = goes_b5_indices[goes_b5_crop_idx]
    goes_b5_data_flat = goes_b5_data_flat[goes_b5_crop_idx]    
    print('gb5 cropped length: ', goes_b5_data_flat.size)

if False:
    print('goes data 0,0 : ',goes_b5_data[0,0])
    nbins = 10
    plt.figure()
    n,bins,patches = plt.hist(goes_b5_data.flatten(),nbins,facecolor='blue',alpha=0.5)
    plt.show()
    exit()

if(False):
    for i in range(28):
        l = 0.5*np.pi*6370.0/np.power(2,i)
        print(i,l,gd.resolution(l))

print('goes_b5_lat.shape:    ',goes_b5_lat.shape)
print('goes_b5_lon.shape:    ',goes_b5_lon.shape)
print('goes_b5_data.shape:   ',goes_b5_data.shape)
print('goes_b5_elemRes (km): ',goes_b5_elemRes)
print('goes_b5_ res(elemRes):',gd.resolution(goes_b5_elemRes))
print('goes_b5_type(lat[0]): ',type(goes_b5_lat[0,0]))

if False:
# Reduce the ROI and check
    i0 = 500*goes_b5_lat1
    n = 10
    goes_b5_lat_flat = goes_b5_lat.flatten()[i0:i0+n]
    # print('type(lat_flat[0]): ',type(lat_flat[0]))
    goes_b5_lon_flat = goes_b5_lon.flatten()[i0:i0+n]
    goes_b5_data_flat = goes_b5_data.flatten()[i0:i0+n]
    goes_b5_indices = np.zeros([n],dtype=np.int64)
    goes_b5_indices = ps.from_latlon(goes_b5_lat_flat,goes_b5_lon_flat,int(gd.resolution(goes_b5_elemRes)))
    goes_b5_indices = ps.from_latlon(goes_b5_lat_flat,goes_b5_lon_flat,int(gd.resolution(goes_b5_elemRes)))
    if False:
        for i in range(n):
            print(i,goes_b5_lat_flat[i],goes_b5_lon_flat[i],hex(goes_b5_indices[i]))

# exit()

print('0 gb5ind type: ',type(goes_b5_indices))
print('0 gb5ind shap: ',goes_b5_indices.shape)
# goes_b5_indices=goes_b5_indices[50*3311:51*3311]
# goes_b5_indices=goes_b5_indices[100*3311:300:3311]
# goes_b5_indices=goes_b5_indices[299*3311:300*3311]
# goes_b5_indices=goes_b5_indices[1000*3311:1100*3311]
# print('1 gb5ind shap: ',goes_b5_indices.shape)
# print('1 gb5ind type: ',type(goes_b5_indices))

m2_g5_matches = []
m2_ar_nroi = m2_ar_indices.size
m2_ar_n0 = 0
m2_ar_n1 = m2_ar_n0 + m2_ar_nroi
# for i in range(m2_ar_n0,m2_ar_n1):
#     print(i,' Calculating cmp for ',hex(m2_ar_indices[i]))
#     cmp = np.nonzero(ps.cmp_spatial([m2_ar_indices[i]],goes_b5_indices))
#     print(i,' cmp size:  ',cmp[0].size)
#     if cmp[0].size > 0:
#         m2_g5_matches.append((i,cmp))
#     # print(i,' cmp type:  ',type(cmp))
#     # print(i,' cmp:       ',cmp)
#     # print(i,' cmp shape: ',cmp[0].shape)

m2_g5_matches = {}
# k=0
for i in range(len(goes_b5_indices)):
    # k=k+1; print('comparison iter k ',k)
    # print('i ',i,type(goes_b5_indices[i]),type(m2_ar_indices))
    cmp = np.nonzero(ps.cmp_spatial(np.array([goes_b5_indices[i]],dtype=np.int64),m2_ar_indices))
    if cmp[0].size > 0:
        for j in cmp[0]:
            if j not in m2_g5_matches.keys():
                m2_g5_matches[j] = np.array([i],dtype=np.int64)
            else:
                m2_g5_matches[j] = np.append(m2_g5_matches[j],i)

print('type(m2_g5_matches): ',type(m2_g5_matches))
# print('cmp len: ',cmp.size)

# exit()

# Visualize

if True:
    proj=ccrs.PlateCarree()
    transf = ccrs.Geodetic()
    
    plt.figure()
    ax = plt.axes(projection=proj)
    ax.set_global()
    # ax.set_xlim(-180,180)
    # ax.set_ylim(-90,90)
    ax.coastlines()

    # Plot m2
    # plt.scatter(m2_long,m2_latg,s=15,c=m2_data,transform=transf)
    plt.scatter(m2_long,m2_latg,s=20,c=m2_data,transform=transf)
    
    # plt.contourf(goes_b5_lon,goes_b5_lat,goes_b5_data,60,transform=transf)
    plt.scatter(goes_b5_lon_flat,goes_b5_lat_flat,s=0.75,c=goes_b5_data_flat,transform=ccrs.Geodetic())
    
    # Plot m2 AR thresh.
    if False:
        # gd.plot_indices(goes_b5_indices,c='g',transform=transf)
        gd.plot_indices(m2_ar_indices[m2_ar_n0:m2_ar_n1],c='c',transform=transf,lw=0.25)
        # Plot matching goes_b5.
        for j in m2_g5_matches:
            gd.plot_indices(goes_b5_indices[m2_g5_matches[j]],c='r',transform=transf,lw=0.25)
    
    # Plot m2
    # plt.scatter(m2_long,m2_latg,s=15,c=m2_data,transform=transf)

    plt.show()

if True:
    plt.figure()
    ax = plt.axes()
    # ax.set_xlim(0,200)
    # ax.set_ylim(0,200)
    
    if False:
        for j in m2_g5_matches:
            x = m2_ar_data[j]
            for y in goes_b5_data_flat[m2_g5_matches[j]]:
                plt.scatter(x,y,s=15,alpha=0.5,c='black')

    if True:
        ndat = len(m2_g5_matches)
        xdat = np.zeros([ndat],dtype=np.double)
        ydat = np.zeros([ndat],dtype=np.double)
        wght = np.zeros([ndat],dtype=np.double)
        k=0
        for j in m2_g5_matches:
            x = m2_ar_data[j]
            # y = 
            ymn = np.amin(goes_b5_data_flat[m2_g5_matches[j]])
            ymx = np.amax(goes_b5_data_flat[m2_g5_matches[j]])
            ystd = np.std(goes_b5_data_flat[m2_g5_matches[j]])
            ybar = np.mean(goes_b5_data_flat[m2_g5_matches[j]])
            xdat[k] = x
            ydat[k] = ybar
            wght[k] = 1.0/ystd
            k=k+1
            plt.plot([x,x],[ymn,ymx],alpha=0.5,c='red')
            plt.plot([x,x],[ybar-ystd,ybar+ystd],alpha=0.5,c='red')
            plt.scatter(x,ybar,s=30,c='darkred')
            plt.scatter([x,x],[ymn,ymx],s=15,c='coral')
        coeffs = np.polyfit(xdat,ydat,1,w=wght)
        print('coeffs: ',coeffs)
        poly   = np.poly1d(coeffs)
        plt.plot(xdat,poly(xdat),c='black')
    plt.show()

# For m2 vs. b5
## # coeffs:  [ -291.49804783 20692.06646311]
## >>> def p(x):
## ...    return -291.5*x + 20692.066
## ... 
## >>> p(2)
## 20109.066
## >>> p(3)
## 19817.566
## >>> p(20)
## 14862.065999999999
## So the threshold for b5 is < ~14800.

exit()

# plt.contourf(m2_long,m2_latg,m2_data,60,transform=ccrs.PlateCarree())
plt.scatter(m2_long,m2_latg,s=15,c=m2_data,transform=transf)
# plt.show()

# plt.contourf(lon,lat,data,60,transform=ccrs.PlateCarree())
# plt.scatter(lon,lat,s=20,c=data)

plt.scatter(goes_b5_lon_flat,goes_b5_lat_flat,s=30,c=goes_b5_data_flat,transform=ccrs.Geodetic())
gd.plot_indices(goes_b5_indices,transform=ccrs.Geodetic())

if True:
    print('Add circles around observation points.')
    for i in range(goes_b5_lon_flat.size):
        lonplot,latplot = gd.bbox_lonlat(goes_b5_lat_flat[i],goes_b5_lon_flat[i],goes_b5_elemRes,close=True)
        ax.plot(lonplot,latplot,True,transform=ccrs.Geodetic())
        ax.add_patch(plt.Circle((goes_b5_lon_flat[i],goes_b5_lat_flat[i]),radius=goes_b5_elemRes*180.0/(np.pi*6371.0),fill=False,transform=ccrs.Geodetic()))

if False:
    i = 2;
    print('Add a "bounding box" around point ',i)
    lonplot,latplot = gd.bbox_lonlat(goes_b5_lat_flat[i],goes_b5_lon_flat[i],goes_b5_elemRes,close=False)
    hull = ps.to_hull_range_from_latlon(latplot,lonplot,13,300)
    # hull = ps.to_hull_range_from_latlon(latplot,lonplot,15,300) # Looks cool!
    gd.plot_indices(hull,c='b',transform=ccrs.Geodetic(),lw=1.5)
    print('Compare the goes_b5_indices with the "hulled bounding box."')
    print('cmp len goes_b5_indices: ',len(goes_b5_indices),goes_b5_indices.shape)
    print('cmp len hull:    ',len(hull),hull.shape)
    cmp = ps.cmp_spatial(goes_b5_indices,hull)
    cmpr = ps.cmp_spatial(hull,goes_b5_indices)
    ngoes_b5_indices = len(goes_b5_indices)
    nhull = len(hull)
    for i in range(ngoes_b5_indices):
        for j in range(nhull):
            if((cmp[i*nhull+j] != 0 or cmpr[j*ngoes_b5_indices+i]) != 0):
                ping = "***"
            else:
                ping = ""
            print(i,j,' i,j,cmp: ',(cmp[i*nhull+j],cmpr[j*ngoes_b5_indices+i]),hex(goes_b5_indices[i]),hex(hull[j]))
    print('cmp Plot the intersecting triangles in a thicker red line.')
    gd.plot_indices(hull[[18,19,20,21]],c='r',transform=ccrs.Geodetic(),lw=2)


gd.plot_indices(m2_indices,c='g',transform=ccrs.Geodetic(),lw=1)

# ax.add_patch(plt.Circle((lon_flat[i],lat_flat[i]),radius=elemRes*180.0/(np.pi*6371.0),color=None,fc=None,ec='c',fill=False))

plt.show()
