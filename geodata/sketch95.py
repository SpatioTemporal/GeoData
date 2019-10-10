
# Fit skew normals to the two correlated groups. M2-AR-Positive and M2-AR-Negative, i.e. > or < 2cm.

# Analyze the difference B5-B4, i.e. 12-11 microns.

import geodata as gd
import h5py as h5
from netCDF4 import Dataset
import numpy as np
import pystare as ps

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

from scipy.stats import norm,skewnorm

from sortedcontainers import SortedDict


workFileName = "work.h5"
#workFile     = h5.File(workPath+workFileName,'r')
workFile     = h5.File(workFileName,'r')

tpw_scale  = workFile['/merra2_description']['tpw_scale']
tpw_offset = workFile['/merra2_description']['tpw_offset']
print('tpw scale offset: ',tpw_scale,tpw_offset)

wf_indices = workFile['/image']['stare_spatial']

b5_img = workFile['/image']['goes_b5']
print('b5 mnmx: ',np.amin(b5_img),np.amax(b5_img))

b4_img = workFile['/image']['goes_b4']
print('b4 mnmx: ',np.amin(b4_img),np.amax(b4_img))

b3_img = workFile['/image']['goes_b3']
print('b3 mnmx: ',np.amin(b3_img),np.amax(b3_img))

b45_img = b4_img - b5_img
# b45_img = b5_img
# b45_img = b4_img
print('b45 mnmx: ',np.amin(b45_img),np.amax(b45_img))

# m2_img = workFile['/image']['merra2_tpw']
m2_img = tpw_offset + tpw_scale*workFile['/image']['merra2_tpw']
print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))

# b5_img_tot = b5_img[np.where(b5_img>1000)]
m2_img_ge2_idx = np.where((m2_img >= 20.0) & (b5_img>1000)) # This is where TPW is high and b5 is low.
m2_img_lt2_idx = np.where((m2_img < 20.0) & (b5_img>1000))  # Reverse.

nx = workFile['/image_description']['nx']
ny = workFile['/image_description']['ny']

crop_lat=(  27.0,  39.0)
crop_lon=(-161.5,-159.5)
crop_lats = np.array([crop_lat[0],crop_lat[0],crop_lat[1],crop_lat[1]],dtype=np.double)
crop_lons = np.array([crop_lon[0],crop_lon[1],crop_lon[1],crop_lon[0]],dtype=np.double)
ar_resolution = 7
ar_cover = ps.to_hull_range_from_latlon(crop_lats,crop_lons,ar_resolution)
print('ar_cover size: ',ar_cover.size)
ar_cover_mn = np.amin(ar_cover)
ar_cover_mx = np.amax(ar_cover)
print('ar_cover mnmx: ',ar_cover_mn,ar_cover_mx)
wf_indices_crop0_idx = np.where( (ar_cover_mn < wf_indices) & (wf_indices < ar_cover_mx) )
wf_indices_crop0 = wf_indices[wf_indices_crop0_idx]
print('wf_indices_crop0 size: ',wf_indices_crop0.size)

# exit()

wf_ar_idx_crop0 = np.full(wf_indices_crop0.shape,False,dtype=np.bool)
# subset_idx = ps.intersect(ar_cover,wf_indices)
ktr = 0
for i in range(wf_ar_idx_crop0.size):
    if ktr % (wf_ar_idx_crop0.size/20) == 0:
        print('ktr: %2d%%'%int(100*ktr/wf_ar_idx_crop0.size),end='\r',flush=True)
    ktr = ktr + 1
    cmp = ps.cmp_spatial(np.array([wf_indices_crop0[i]]),ar_cover) # Note cmp_spatial can be sped up.
    wf_ar_idx_crop0[i] = (1 in cmp) or (-1 in cmp)
print('ktr: done')

# wf_ar_idx = wf_indices_crop0[wf_ar_idx_crop0]
wf_ar_idx = wf_indices[wf_indices_crop0_idx][wf_ar_idx_crop0]
b3_ar   =  b3_img[wf_indices_crop0_idx][wf_ar_idx_crop0]
b4_ar   =  b4_img[wf_indices_crop0_idx][wf_ar_idx_crop0]
b5_ar   =  b5_img[wf_indices_crop0_idx][wf_ar_idx_crop0]
b45_ar  = b45_img[wf_indices_crop0_idx][wf_ar_idx_crop0]
m2_ar   =  m2_img[wf_indices_crop0_idx][wf_ar_idx_crop0]

# wf_ar_idx = [\
#              (1 in ps.cmp_spatial(np.array([wf_ar_idx[i]]),ar_cover))\
#              or (-11 in ps.cmp_spatial(np.array([wf_ar_idx[i]]),ar_cover))\
#              for i in range(wf_ar_idx.size)]

class group_value(object):
    def __init__(self):
        self.m2 = []
        self.b3 = []
        self.b4 = []
        self.b5 = []
        self.b45 = []
        return
        
    def add(self,m2,b3,b4,b5,b45,i):
        if m2[i] not in self.m2:
            self.m2.append(m2[i])
        self.b3.append(b3[i])
        self.b4.append(b4[i])
        self.b5.append(b5[i])
        self.b45.append(b45[i])
        return

# print('')
# print('wf_indices_crop0 len: ',len(wf_indices_crop0),wf_indices_crop0.shape)
# print('wf_ar_idx len:        ',len(wf_ar_idx),wf_ar_idx.shape)
# print('m2_ar len:            ',len(m2_ar),m2_ar.shape)
# print('b45_ar len:           ',len(b45_ar),b45_ar.shape)
# print('')
group = SortedDict()
ktr = 0
for i in range(wf_ar_idx.size):
    if ktr % (wf_ar_idx_crop0.size/20) == 0:
        print('ktr: %2d%%'%int(100*ktr/wf_ar_idx_crop0.size),end='\r',flush=True)
    ktr = ktr + 1
    # sid0 = wf_indices_crop0[i]
    sid0 = wf_ar_idx[i]
    sid  = gd.spatial_clear_to_resolution(gd.spatial_coerce_resolution(sid0,ar_resolution))
    if sid not in group.keys():
        group[sid] = group_value()
    group[sid].add(m2_ar,b3_ar,b4_ar,b5_ar,b45_ar,i)
    if m2_ar[i] not in group[sid].m2:
        group[sid].m2.append(m2_ar[i])
    group[sid].b45.append(b45_ar[i])
print('ktr: done')

x=[]
y=[]
w=[]

for i in group:
    group[i].b45 = [np.mean(group[i].b4) - np.mean(group[i].b5)]
    x.append(group[i].m2[0])
    y.append(group[i].b45[0])
    w5 = np.std(group[i].b5)
    w4 = np.std(group[i].b4)
    w.append(1.0/np.sqrt(w4*w4+w5*w5))

x = np.array(x)
y = np.array(y)
w = np.array(w)

print('w  ',type(w),w.shape,w.size,w.dtype)
print('x  ',type(x),x.shape,x.size,w.dtype)

coeffs = np.polyfit(x,y,1,w=w)
poly   = np.poly1d(coeffs)

plt.figure()
plt.scatter(x,y)
plt.plot(x,poly(x),c='black')
plt.show()
# exit()

# fig,axs = plt.subplots(nrows=1)
# axs.scatter(m2_img,b45_img,s=4,alpha=0.5)
# plt.show()

plt.figure()
xdat = np.zeros(len(group.keys()))
ybar = np.zeros(len(group.keys()))
ystd = np.zeros(len(group.keys()))
ymn  = np.zeros(len(group.keys()))
ymx  = np.zeros(len(group.keys()))
k = 0
for i in group:    
    if len(group[i].m2) > 1:
        print(k,' note: ',i,group[i].m2,len(group[i].b45))
    y = np.array(group[i].b4)-np.array(group[i].b5)
    ybar[k]=np.mean(y)
    ystd[k]=np.std(y)
    ymn[k] =np.amin(y)
    ymx[k] =np.amax(y)
    x0 = group[i].m2[0]
    x  = np.full(y.shape,x0)
    xdat[k] = x0
    # print('x ',x.size,type(x),x.dtype)
    # print('y ',y.size,type(y),y.dtype)
    # plt.scatter(x,y,s=4,alpha=0.5)
    # plt.plot([x0,x0],[ybar[k]-ystd[k],ybar[k]+ystd[k]],alpha=0.5,c='red')    
    plt.scatter(x0,ybar[k],s=30,c='darkred')
    plt.scatter([x0,x0],[ymn[k],ymx[k]],s=15,c='coral')
    plt.plot([x0,x0],[ymn[k],ymx[k]],alpha=0.5,c='red')
    k=k+1

plt.plot(xdat,poly(xdat),c='black')
plt.show()
exit()

plt.figure()
plt.scatter(m2_ar,b45_ar,s=4,alpha=0.5)
plt.show()

projection = ccrs.PlateCarree()
transform  = ccrs.Geodetic()

plt.figure()
ax = plt.axes(projection=projection)
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.coastlines()

if False:
    lons,lats,intmat = ps.triangulate_indices(wf_indices_crop0)
    # lons,lats,intmat = ps.triangulate_indices(subset_idx)
    triang = tri.Triangulation(lons,lats,intmat)
    plt.triplot(triang,'b-',transform=transform,lw=1.5,markersize=3)

if True:
    lons,lats,intmat = ps.triangulate_indices(wf_ar_idx)
    # lons,lats,intmat = ps.triangulate_indices(subset_idx)
    triang = tri.Triangulation(lons,lats,intmat)
    plt.triplot(triang,'b-',transform=transform,lw=1.5,markersize=3)


lons,lats,intmat = ps.triangulate_indices(ar_cover)
# lons,lats,intmat = ps.triangulate_indices(subset_idx)
triang = tri.Triangulation(lons,lats,intmat)
plt.triplot(triang,'r-',transform=transform,lw=1.5,markersize=3)

plt.show()

fig,axs = plt.subplots(ncols=2,subplot_kw={'projection':projection})
for iaxs in range(len(axs)):
    axs[iaxs].get_xaxis().set_visible(False)
    axs[iaxs].get_yaxis().set_visible(False)
b45_lats,b45_lons = ps.to_latlon(wf_ar_idx)
axs[0].set_title('M2')
axs[0].scatter(b45_lons,b45_lats,c=m2_ar,s=10,transform=transform)
axs[1].set_title('B5-B4')
axs[1].scatter(b45_lons,b45_lats,c=b45_ar,s=10,transform=transform)
plt.show()

fig,axs = plt.subplots(nrows=4)
# for iaxs in len(axs):
# if False:
#     iaxs = 0
#     axs[iaxs].get_xaxis().set_visible(False)
#     axs[iaxs].get_yaxis().set_visible(False)
#     axs[iaxs].scatter(m2_img,b45_img)
# else:
iaxs = 0
axs[iaxs].get_xaxis().set_visible(False)
axs[iaxs].get_yaxis().set_visible(False)
axs[iaxs].imshow(m2_img.reshape(ny,nx))

iaxs = 1
axs[iaxs].get_xaxis().set_visible(False)
axs[iaxs].get_yaxis().set_visible(False)
axs[iaxs].imshow(b45_img.reshape(ny,nx))

iaxs = 2
axs[iaxs].get_xaxis().set_visible(False)
axs[iaxs].get_yaxis().set_visible(False)
axs[iaxs].imshow(b5_img.reshape(ny,nx))

iaxs = 3
axs[iaxs].get_xaxis().set_visible(False)
axs[iaxs].get_yaxis().set_visible(False)
axs[iaxs].imshow(b4_img.reshape(ny,nx))

plt.show()


### ### FIGURES ### 
### fig,axs = plt.sub
### 
### plt.figure()
### plt.imshow(b5_img.reshape(nx,ny))
### plt.show()
### 
### b5_img_ge = b5_img.copy()
### b5_img_lt = b5_img.copy()
### 
### b5_img_ge[m2_img_lt2_idx]=0
### b5_img_lt[m2_img_ge2_idx]=0
### 
### plt.imshow(b5_img_ge.reshape(nx,ny))
### plt.show()
### 
### plt.imshow(b5_img_lt.reshape(nx,ny))
### plt.show()

# b45_img_ge = b45_img[m2_img_ge2_idx]
# b45_img_lt = b45_img[m2_img_lt2_idx]

#

exit()

# Fit Gaussians here... Then can do a likelihood ratio...

low  =  2000
high = 22000
nbins_ = 20
bins_ = low + np.arange(nbins_)*(high-low)/nbins_

print('bins_',bins_)
nsum = 5
bins_1 = [bins_[0]] +[ i for i in bins_[nsum:]]
bins_ = bins_1

#+++    nx   = len(x)
print('bins_',bins_)

# fig, axs = plt.subplots(nrows=4)
# n,bins,patches = axs[0].hist(b5_img_tot,bins=bins_,facecolor='blue',alpha=0.2)
# n,bins,patches = axs[1].hist(b5_img_ge,bins=bins_,facecolor='cyan',alpha=0.2)
# n,bins,patches = axs[2].hist(b5_img_lt,bins=bins_,facecolor='green',alpha=0.2)
# n,bins,patches = axs[3].hist(m2_img,nbins,facecolor='red',alpha=0.5)
# plt.show()

fig,ax1 = plt.subplots()

ax1.set_title('GOES B5 decomposed against M2-TPW>2')
# ax1.annotate("b5[m2>2]/tot.",xy=(2000,500000))
# ax1.annotate("b5[m2<2]/tot.", xy=(2000,250000))

ax1.set_xlabel('B5 Raw Counts')
ax1.set_ylabel('Histogram')

alpha=0.6
n,bins_tot,patches_tot = ax1.hist(b5_img_tot,bins=bins_,facecolor='blue',alpha=alpha)
n,bins_ge,patches_ge   = ax1.hist(b5_img_ge,bins=bins_,facecolor='cyan',alpha=alpha)
n,bins_lt,patches_lt   = ax1.hist(b5_img_lt,bins=bins_,facecolor='red',alpha=alpha)

y_hist_ge = [ i.get_height() for i in patches_ge ]
y_hist_lt = [ i.get_height() for i in patches_lt ]

# print('patches_lt: ',patches_lt)
# print('patches_lt: ',[h.get_height() for h in patches_lt])


bkg = 90000

ax2 = ax1.twinx()
if True:
    x = [ 0.5*(bins_tot[i]+bins_tot[i+1]) for i in range(len(bins_tot)-1) ]
    y = [ (patches_lt[i].get_height())/(patches_ge[i].get_height()) for i in range(len(patches_ge)) ]
    ax2.plot(x,y,color='darkorange',linewidth=3)
    

if False:
    # ax2.set_ylim(0,2.5)
    x = [ 0.5*(bins_tot[i]+bins_tot[i+1]) for i in range(len(bins_tot)-1) ]
    y = [ patches_ge[i].get_height()/patches_lt[i].get_height() for i in range(len(patches_ge)) ]
    # y = [ patches_lt[i].get_height()/patches_ge[i].get_height() for i in range(len(patches_ge)) ]
    # y = np.array([ patches_lt[i].get_height()/patches_ge[i].get_height() for i in range(len(patches_ge)) ])
    # y = np.array([ np.log(patches_lt[i].get_height())/np.log(patches_ge[i].get_height()) for i in range(len(patches_ge)) ])

#+++    nx   = len(x)
#+++    x = np.array(x)
#+++    nsum = 8
#+++    xbar   = np.sum(x[0:nsum])/float(nsum)
#+++    ytilde = np.sum(np.array([patches_lt[i].get_height() for i in range(nsum)]))/np.sum(np.array([patches_ge[i].get_height() for i in range(nsum)]))
#+++
#+++    x1 = np.zeros([nx-nsum+1])
#+++    y1 = np.zeros([nx-nsum+1])
#+++
#+++    x1[0]=xbar;   x1[1:]=x[nsum:]
#+++    y1[0]=ytilde; y1[1:]=y[nsum:]
#+++    ax2.plot(x1,y1,color='blue')

    ax2.plot(x,y,color='blue')
    ax2.plot([x[0],x[-1]],[1,1],color='grey')

ax2.plot([x[0],x[-1]],[1,1],color='grey')
ax2.set_ylabel('b5 m2 on/off ratio')

x = np.linspace(2000,21500,100)

# mu_ge,std_ge = norm.fit(b5_img_ge)
# p_ge = norm.pdf(x,mu_ge,std_ge)

a,loc,scale = skewnorm.fit(b5_img_ge[np.where(b5_img_ge>15000)])
p_ge = skewnorm.pdf(x,a,loc,scale)
p_ge = p_ge/np.amax(p_ge)
p_ge = p_ge*(np.amax(y_hist_ge)-bkg)
ax1.plot(x,p_ge+bkg,linewidth=2,color='cyan')

# mu_lt,std_lt = norm.fit(b5_img_lt)
# p_lt = norm.pdf(x,mu_lt,std_lt)

a,loc,scale = skewnorm.fit(b5_img_lt[np.where((b5_img_lt>7500) & (b5_img_lt<15000))])
p_lt = skewnorm.pdf(x,a,loc,scale)
p_lt = p_lt/np.amax(p_lt)
p_lt = p_lt*(np.amax(y_hist_lt)-bkg)
ax1.plot(x,p_lt+bkg,linewidth=2,color='gold')

ax1.plot(x,p_lt+p_ge+2*bkg,linewidth=2,color='lime')

# ratio
ax2.plot(x,(p_lt+bkg)/(p_ge+bkg),linewidth=3,color='orange')

plt.show()







