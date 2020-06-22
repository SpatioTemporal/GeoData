# GeoData

Exercises in Geoscience Data Integration using SpatioTemporal Adaptive Resolution Encoding.

For more information contact Michael Rilee (mike@rilee.net).

The key modules in this Python 3 package are as follows.
- geodata.py
- join_goes_merra2.py
- modis_coarse_to_fine_geolocation
- stopwatch.py

# geodata.py
Provides important data access and processing functions.

# join_goes_merra2.py
Reads and joins GOES and MERRA-2 data and writes to hdf5 if required.

# modis_coarse_to_fine_geolocation
Aids geolocation of MODIS data.

# stopwatch.py
Provides timing and logging functions.

# NOTES.org
Contains notes about the sketches in geodata. 

# Dependencies
STARE
- github.com/SpatioTemporal/STARE
- github.com/SpatioTemporal/pystare

Python 3
- SortedContainers, h5py, netCDF4
- And more...

# Installation

    pip3 install git+git://github.com/MichaelLeeRilee/GeoData.git

# Acknowledgments

Work supported in part by NASA ACCESS-17.

Copyright 2019, Mike Rilee, Rilee Systems Technologies LLC.


