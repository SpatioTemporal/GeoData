
import cartopy.crs as ccrs
import geodata as gd
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pystare as ps


def main():
    print('main')

    indices = ps.from_latlon([-45,45],[45,45],1)
    print('level zero: ',indices)
    latv,lonv,latc,lonc = ps.to_vertices_latlon(indices)
    print('  latv:     ',latv)
    print('  lonv:     ',lonv)
    print('  latc:     ',latc)
    print('  lonc:     ',lonc)

    if True:
        proj=ccrs.PlateCarree()
        transf = ccrs.Geodetic()
        plt.figure()
        ax = plt.axes(projection=proj)
        ax.set_global()
        ax.coastlines()

        gd.plot_indices(ps.from_latlon(np.arange(-89,89,dtype=np.float),-2*np.arange(-89,89,dtype=np.float),0),c='g',transform=transf,lw=0.25)
        # gd.plot_indices(ps.from_latlon([0,0,15],[-15,0,15],1),c='g',transform=transf,lw=0.25)
        # gd.plot_indices(ps.from_latlon([0,0,15],[-15,0,15],2),c='r',transform=transf,lw=0.25)
        # gd.plot_indices(ps.from_latlon([-45,45],[45,45],0),c='r',transform=transf,lw=0.25)
        # gd.plot_indices(ps.from_latlon([-45],[45],0),c='r',transform=transf,lw=0.25)


        plt.show()
        return


if __name__ == '__main__':
    main()
