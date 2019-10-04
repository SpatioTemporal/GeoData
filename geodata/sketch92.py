
# Fit skew normals to the two correlated groups. M2-AR-Positive and M2-AR-Negative, i.e. > or < 2cm.

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

workFileName = "work.h5"
#workFile     = h5.File(workPath+workFileName,'r')
workFile     = h5.File(workFileName,'r')

tpw_scale  = workFile['/merra2_description']['tpw_scale']
tpw_offset = workFile['/merra2_description']['tpw_offset']
print('tpw scale offset: ',tpw_scale,tpw_offset)

b5_img = workFile['/image']['goes_b5']
print('b5 mnmx: ',np.amin(b5_img),np.amax(b5_img))

# m2_img = workFile['/image']['merra2_tpw']
m2_img = tpw_offset + tpw_scale*workFile['/image']['merra2_tpw']
print('m2 mnmx: ',np.amin(m2_img),np.amax(m2_img))


b5_img_tot = b5_img[np.where(b5_img>1000)]
m2_img_ge2_idx = np.where((m2_img >= 20.0) & (b5_img>1000))
m2_img_lt2_idx = np.where((m2_img < 20.0) & (b5_img>1000))

b5_img_ge = b5_img[m2_img_ge2_idx]
b5_img_lt = b5_img[m2_img_lt2_idx]

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
ax1.set_xlabel('B5 Raw Counts')
ax1.set_ylabel('Histogram')
ax1.annotate("(b5 m2off)/total",xy=(2250,500000))
ax1.annotate("(b5 m2on)/total", xy=(2250,250000))

n,bins_tot,patches_tot = ax1.hist(b5_img_tot,bins=bins_,facecolor='blue',alpha=0.2)
n,bins_ge,patches_ge   = ax1.hist(b5_img_ge,bins=bins_,facecolor='cyan',alpha=0.2)
n,bins_lt,patches_lt   = ax1.hist(b5_img_lt,bins=bins_,facecolor='green',alpha=0.2)

y_hist_ge = [ i.get_height() for i in patches_ge ]
y_hist_lt = [ i.get_height() for i in patches_lt ]

# print('patches_lt: ',patches_lt)
# print('patches_lt: ',[h.get_height() for h in patches_lt])


ax2 = ax1.twinx()
if True:
    x = [ 0.5*(bins_tot[i]+bins_tot[i+1]) for i in range(len(bins_tot)-1) ]
    y = [ patches_ge[i].get_height()/patches_tot[i].get_height() for i in range(len(patches_ge)) ]

    ax2.plot(x,y,color='red')
    # print(bins_tot)
    # print(bins_ge)
    # print(bins_lt)
    
    x = [ 0.5*(bins_tot[i]+bins_tot[i+1]) for i in range(len(bins_tot)-1) ]
    y = [ patches_lt[i].get_height()/patches_tot[i].get_height() for i in range(len(patches_ge)) ]

    ax2.plot(x,y,color='blue')

if False:
    # ax2.set_ylim(0,2.5)
    x = [ 0.5*(bins_tot[i]+bins_tot[i+1]) for i in range(len(bins_tot)-1) ]
    y = [ patches_lt[i].get_height()/patches_ge[i].get_height() for i in range(len(patches_ge)) ]
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

ax2.set_ylabel('g5 m2 on/off ratio')

x = np.linspace(2000,21500,100)

# mu_ge,std_ge = norm.fit(b5_img_ge)
# p_ge = norm.pdf(x,mu_ge,std_ge)

bkg = 90000

a,loc,scale = skewnorm.fit(b5_img_ge[np.where(b5_img_ge>15000)])
p_ge = skewnorm.pdf(x,a,loc,scale)
p_ge = p_ge/np.amax(p_ge)
p_ge = p_ge*(np.amax(y_hist_ge)-bkg)
ax1.plot(x,p_ge+bkg,linewidth=2)

# mu_lt,std_lt = norm.fit(b5_img_lt)
# p_lt = norm.pdf(x,mu_lt,std_lt)

a,loc,scale = skewnorm.fit(b5_img_lt[np.where((b5_img_lt>7500) & (b5_img_lt<15000))])
p_lt = skewnorm.pdf(x,a,loc,scale)
p_lt = p_lt/np.amax(p_lt)
p_lt = p_lt*(np.amax(y_hist_lt)-bkg)
ax1.plot(x,p_lt+bkg,linewidth=2)

ax1.plot(x,p_lt+p_ge+2*bkg,linewidth=2)

plt.show()







