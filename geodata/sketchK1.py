
# Load MODIS and construct an h5 file.

import numpy as np
from pyhdf.SD import SD, SDC
# import pprint

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs
import datetime
import geodata as gd
import pystare as ps
import h5py as h5
from netCDF4 import Dataset

from sketchK0 import with_hdf_get

###########################################################################

def main():

    viz = True

    if viz:
        proj   = ccrs.PlateCarree()
        transf = ccrs.Geodetic()
        nrows = 1
        fig, axs = plt.subplots(nrows=nrows,subplot_kw={'projection':proj,'transform':transf})
        if nrows == 1:
            axs=[axs]

        ax = axs[0]
        an = ax.annotate(
             'Figure:  sketchK1\n'
            +'Date:    %s\n'%datetime.date.today()
            +'Version: 2020-0407-1\n'
            +'Author:  M. Rilee, RSTLLC\n'
            +'Email:   mike@rilee.net\n'
            ,xy=(0.7,0.025)
            ,xycoords='figure fraction'
            ,family='monospace'
        )
        
        plt.show()

    print('done')
    return

###########################################################################

if __name__ == '__main__':
    main()
