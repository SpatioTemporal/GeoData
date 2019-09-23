
from math import ceil
import pystare as ps

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy as np

class track(object):
    coeffs=[]
    def __init__(self,coeffs=[],dxmnmx=[],xyc=[],rthick=1):
        self.coeffs=coeffs
        self.xyc=xyc
        if(len(xyc) == 2):
            self.x0 = xyc[0]
            self.y0 = xyc[1]
        else:
            self.x0 = 0
            self.y0 = 0
        xmnmx=[]
        if(len(dxmnmx) == 2):
            xmnmx = [0,0]
            if(len(xyc) == 2):
                xmnmx[0] += xyc[0]; xmnmx[1] += xyc[0]
            xmnmx[0] += dxmnmx[0];xmnmx[1] += dxmnmx[1];
        self.xmnmx=xmnmx
        self.rthick=rthick

    def evaluate(self,x,y):
        # print('x.shape',x.shape); print('y.shape',y.shape)
        xc=np.unique(np.copy(x))
        if len(self.xmnmx) == 2:
            xc = xc[np.where(xc >= self.xmnmx[0])]
            xc = xc[np.where(xc <= self.xmnmx[1])]
        yc=np.zeros(xc.shape)
        # print('xc.shape',xc.shape); print('yc.shape',yc.shape)
        n=len(self.coeffs)
        for i in range(n-1,-1,-1):
            # print(i,self.coeffs[i])
            yc=self.coeffs[i]+yc*(xc - self.x0)
        yc += self.y0
        # return yc
        vsum=np.zeros(x.shape)
        for i in range(x.size):
            for k in range(xc.size):
                deltax = x[i]-xc[k]
                deltay = y[i]-yc[k]
                deltar = np.sqrt(deltax*deltax + deltay*deltay)
                deltav = np.exp(-deltar/self.rthick)
                # deltav = deltar
                vsum[i] += deltav

        return vsum
#        y=0*x+coeffs[0]

# nx=100; ny=100
# nx=20; ny=20
# nx=40; ny=10
nx=15; ny=15
# nx=10; ny=10
x=np.linspace(-180,180,nx); y=np.linspace(-90,90,ny)
xg,yg = np.meshgrid(x,y)
xg_flat = xg.flatten(); yg_flat = yg.flatten()

angleScale = 360.0/nx
resolution = ceil(-np.log2(angleScale/90.0))
sid = ps.from_latlon(yg_flat,xg_flat,resolution)
sidg = sid.reshape(xg.shape)

# print('sid'); print(type(sid))
# print(['{:02x}'.format(i) for i in sid[0:10]])
# print(['{:02x}'.format(i) for i in sid])

# t0=track([0,1],rthick=10.0,xmnmx=[-10,30])
# t0=track([5,0.5,0.005,-0.0005,-0.000001],rthick=10.0,dxmnmx=[-30,45])
# t0=track([5,0.5,0.005,-0.0005,-0.000001],rthick=10.0,xyc=[10,20],dxmnmx=[-30,45])
# t0=track([5],rthick=10.0,dxmnmx=[-30,45])
# t0=track([5],rthick=10.0,xyc=[0,45],dxmnmx=[0,90])
# t0=track([5],rthick=10.0,xyc=[0,-45],dxmnmx=[0,90])
# t0=track([5],rthick=10.0,xyc=[-90,-45],dxmnmx=[0,90])
# t0=track([5],rthick=10.0,xyc=[-90,45],dxmnmx=[0,90])
# t0=track([0,1],rthick=10.0,xyc=[-90,45],dxmnmx=[0,90])
# t0=track([0,-1],rthick=10.0,xyc=[-90,45],dxmnmx=[0,90])
# t0=track([0,0,-1.0/90.0],rthick=10.0,xyc=[0,0],dxmnmx=[-90,90])
# t0=track([0,0,-1.0/90.0],rthick=10.0,xyc=[0,90],dxmnmx=[-90,90])
# t0=track([0,0,-1.0/90.0],rthick=10.0,xyc=[0,45],dxmnmx=[-90,90])
# t0=track([5],rthick=10.0,xyc=[0,0],dxmnmx=[-90,90])
# t0=track([5],rthick=10.0,xyc=[0,-45],dxmnmx=[-90,90])
# t0=track([0,1],rthick=10.0,xyc=[0,45],dxmnmx=[-90,90])
# t0=track([0,1],rthick=10.0,xyc=[45,0],dxmnmx=[-90,90])

# thicker
t0=track([0,0,-1.0/90.0],rthick=30.0,xyc=[0,45],dxmnmx=[-90,90])
t1=track([0,0,-1.0/90.0],rthick=15.0,xyc=[0,45],dxmnmx=[-90,90])
t2=track([0,0,-1.0/90.0],rthick= 7.5,xyc=[0,45],dxmnmx=[-90,90])
t3=track([0,0,-1.0/90.0],rthick=12.0,xyc=[0,45],dxmnmx=[-90,90])
    
# print('xg'); print(xg); print('yg'); print(yg)
# print('xg_flat'); print(xg_flat); print('yg_flat'); print(yg_flat)

v0_flat=t0.evaluate(xg_flat,yg_flat)
v0g    = v0_flat.reshape(xg.shape)
v0g_mx = np.max(v0g)
v0g_mn = np.min(v0g)
v0g /= v0g_mx
# v0g_mn = np.min(v0g); v0g_mx = np.max(v0g)
# print('vgmnmx: ',v0g_mn,' ',v0g_mx)

v1_flat=t1.evaluate(xg_flat,yg_flat)
v1g    = v1_flat.reshape(xg.shape)
v1g_mx = np.max(v1g)
v1g_mn = np.min(v1g)
v1g /= v1g_mx

v2_flat=t2.evaluate(xg_flat,yg_flat)
#+ pack check
# for i in range(v2_flat.size):
#   v2_flat[i] = i
#- pack check
v2g    = v2_flat.reshape(xg.shape)
v2g_mx = np.max(v2g)
v2g_mn = np.min(v2g)
v2g /= v2g_mx

v3_flat=t3.evaluate(xg_flat,yg_flat)
v3g    = v3_flat.reshape(xg.shape)
v3g_mx = np.max(v3g)
v3g_mn = np.min(v3g)
v3g /= v3g_mx

# for i in range(0,xg_flat.size):
#    print((xg_flat[i],yg_flat[i],v_flat[i]))

vg = v0g

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.coastlines()
plt.imshow(vg,extent=[-180,180,-90,90],origin='center')
# plt.contourf(xg,yg,v0g,60,transform=ccrs.PlateCarree())
# plt.scatter(xg_flat,yg_flat,s=300,c=v_flat)

plt.show()

# scale
toIntScale  = 1024
toIntOffset = 0
v0gs = np.asarray(v0g*v0g_mx*toIntScale + toIntOffset,dtype=np.int64)
v1gs = np.asarray(v1g*v1g_mx*toIntScale + toIntOffset,dtype=np.int64)
v2gs = np.asarray(v2g*v2g_mx*toIntScale + toIntOffset,dtype=np.int64)
v3gs = np.asarray(v3g*v3g_mx*toIntScale + toIntOffset,dtype=np.int64)
thresholdB0 = 1000

print('<header>')
print('ARMO Synthetic Data Sketch2 Alternative Dataset')
print('v0g-max',v0g_mx)
print('v1g-max',v1g_mx)
print('v2g-max',v2g_mx)
print('v3g-max',v3g_mx)
print('ij-packing','C-style row-major order k=j+i*ny')
print('(nx,ny)',(nx,ny))
print('toIntScale',toIntScale)
print('toIntOffset',toIntOffset)
print('threshold-B0',thresholdB0)
print('<STARE-spatial/> <STARE-temporal/> <Image IJ/> <Band0/> <Band1/> <Band2/> <AR-MASK/> <TPW IJ/> <TPW/>')
# print('hex(sidg[i,j]),hex(time),ijpacked,v0gs[i,j],v1gs[i,j],v2gs[i,j], Band0.ge.'+str(thresholdB0)+',ijpackedTPW,v3gs[i,j]')
print('((9*"%i,")[:-1]%(sidg[i,j],time,ijpacked,v0gs[i,j],v1gs[i,j],v2gs[i,j],Band0.ge.str(thresholdB0),ijpackedTPW,v3gs[i,j]')
print('</header>')
print('<data>')
# for i in range(5,10):
#     for j in range(5,10):
for i in range(nx):
    for j in range(ny):
        ijpacked = j+i*ny # C-style row-major, i=row, j=column, 'flatten' default
        ijpackedTPW = ijpacked # TODO fix
        time=0
        # format-0 print(hex(sidg[i,j]),hex(time),ijpacked,v0gs[i,j],v1gs[i,j],v2gs[i,j],int(v0gs[i,j]>thresholdB0),ijpackedTPW,v3gs[i,j]);
        # format-1
        print((9*"%i,")[:-1]%(sidg[i,j],time,ijpacked,v0gs[i,j],v1gs[i,j],v2gs[i,j],int(v0gs[i,j]>thresholdB0),ijpackedTPW,v3gs[i,j]))
        #+ pack check print(hex(sidg[i,j]),hex(0),ijpacked,i,j,v0gs[i,j],v1gs[i,j],v2g[i,j],v2_flat[ijpacked],v2g[i,j]-v2_flat[ijpacked]) #- pack check
print('</data>')

vg[:,:] = v0gs[:,:]
vg[np.where(vg < thresholdB0)] = 0;

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.coastlines()
plt.imshow(vg,extent=[-180,180,-90,90],origin='center')
# plt.contourf(xg,yg,v0g,60,transform=ccrs.PlateCarree())
# plt.scatter(xg_flat,yg_flat,s=300,c=v_flat)

plt.show()
