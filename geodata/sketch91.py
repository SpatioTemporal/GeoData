import geodata as gd
import h5py as h5
from netCDF4 import Dataset
import numpy as np
import pystare as ps

goes_b5_dataPath   = "/home/mrilee/data/"
goes_b5_dataFile = "goes10.2005.349.003015.BAND_05.nc"
goes_b5_fqFilename = goes_b5_dataPath+goes_b5_dataFile
goes_b5_ds = Dataset(goes_b5_fqFilename)

print('goes temporal id: ',hex(gd.goes10_img_stare_time(goes_b5_ds)[0]))
print([hex(i) for i in ps.from_latlon(goes_b5_ds['lat'][0:4,0:4].flatten(),goes_b5_ds['lon'][0:4,0:4].flatten(),int(gd.resolution(goes_b5_ds['elemRes'][0])))])
print(goes_b5_ds['lat'][0:4,0:4].flatten())
