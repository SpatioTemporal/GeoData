import geodata as gd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs
import pystare as ps

import datetime


def main():

    # proj   = ccrs.PlateCarree()
    proj   = ccrs.Mollweide()
    transf = ccrs.Geodetic()

    plt.figure()
    ax = plt.axes(projection=proj)
    # ax.set_xlim(-180,180)
    # ax.set_ylim(-90,90)
    ax.stock_img()
    ax.coastlines()
    ax.gridlines()

    lats = np.array([-45,-45,-45,-45,-45, -45, 45,45,45, 45, 45, 45],dtype=np.double)
    lons = np.array([  0, 45, 90,135,180, 225,  0,45,90,135,180,225],dtype=np.double)

    tmp_cover = ps.from_latlon(lats,lons,0)
    print('tmp_cover len: ',len(tmp_cover))
    print('tmp_cover:     ',tmp_cover)
    clons,clats,cintmat = ps.triangulate_indices(tmp_cover)
    ctriang = tri.Triangulation(clons,clats,cintmat)
    ax.triplot(ctriang,'b-',transform=transf,lw=1.0,markersize=3,alpha=1.0)    

    zlatv = np.array([
        0, 90, 0,  0,  90,   0,   0,   90,   0,   0,   90,   0,
        0,-90, 0,  0, -90,   0,   0,  -90,   0,   0,  -90,   0
                  ],dtype=np.double)
    zlonv = np.array([
        0,90,90, 90, 90, 180, 180, 270, 270, 270, 360, 360,
        0,90,90, 90, 90, 180, 180, 270, 270, 270, 360, 360
                  ],dtype=np.double)
    zlons,zlats,zintmat = gd.triangulate1(zlatv,zlonv)
    ztriang = tri.Triangulation(zlons,zlats,zintmat)
    ax.triplot(ztriang,'r-',transform=transf,lw=1.0,markersize=3,alpha=1.0)

    an = ax.annotate(
         'Figure:  sketchM0\n'
        +'Date:    %s\n'%datetime.date.today()
        +'Version: 2020-0407-1\n'
        +'Author:  M. Rilee, RSTLLC\n'
        +'Email:   mike@rilee.net\n'
        ,xy=(0.7,0.025)
        ,xycoords='figure fraction'
        ,family='monospace'
        )

    plt.show()

    return

if __name__ == '__main__':
    main()

