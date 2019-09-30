
import geodata as gd
from netCDF4 import Dataset
import numpy as np
import pystare as ps

import os, fnmatch

dataPath   = "/home/mrilee/data/"

listOfFiles = os.listdir(dataPath)
patterns = ["*.nc","*.nc4"]
for pattern in patterns:
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry,pattern):
            print(entry)

print('')
patterns = ["MERRA*.nc4","goes*.nc"]
for pattern in patterns:
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry,pattern):
            tid = gd.temporal_id_from_file(dataPath,entry)
            print(entry,hex(tid),gd.datetime_from_stare(tid))


