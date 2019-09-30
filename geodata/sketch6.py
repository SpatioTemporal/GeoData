from netCDF4 import Dataset
import numpy as np
import pystare as ps

# from geodata import merra2_parse_begin_date, merra2_make_time, merra2_stare_time, format_time, datetime_from_stare
import geodata as gd


### MERRA-2 Simplification

dataPath   = "/home/mrilee/data/"
dataFile   = "MERRA2_300.tavg1_2d_slv_Nx.20051215.nc4"
fqFilename = dataPath+dataFile
ds         = Dataset(fqFilename)

tim = ds['time']
print('tim:                  ',tim)
print('tim units:            ',tim.units)
print('tim begin data:       ',tim.begin_date)
print('tim m2 prs date:      ',gd.merra2_parse_begin_date(tim.begin_date))
print('tim begin data typ:   ',type(tim.begin_date))
print('tim begin time:       ',tim.begin_time)
# print('tim[:]: ',tim[:])

# tid = gd.merra2_stare_time(ds,0)
tid_center = gd.merra2_stare_time(ds)
tid_centers = gd.datetime_from_stare(tid_center)

tid_interval = gd.merra2_stare_time(ds,centered=False)
tid_intervals = gd.datetime_from_stare(tid_interval)

if False:
    print('tid_center:   ',[hex(i) for i in tid_center])
    for i in range(24):
        print('tid_center/s,tid_interval/s:  ',hex(tid_center[i]),tid_centers[i],hex(tid_interval[i]),tid_intervals[i])

### GOES IMG BAND_05

goes_b5_dataPath = "/home/mrilee/data/"
goes_b5_dataFile = "goes10.2005.349.003015.BAND_05.nc"
goes_b5_fqFilename = goes_b5_dataPath+goes_b5_dataFile
goes_b5_ds = Dataset(goes_b5_fqFilename)

# Is the time at the beginning or the middle of the scan? Assume center to make progress.
goes_b5_tid_center = gd.goes10_img_stare_time(goes_b5_ds)
goes_b5_tid_centers = gd.datetime_from_stare(goes_b5_tid_center)
for i in range(goes_b5_tid_center.size):
    print(i,' g5 tid: ',hex(goes_b5_tid_center[i]),goes_b5_tid_centers[i])

print (' g5 tid: ',gd.datetime_from_stare(goes_b5_tid_center[i]))

###

print(ps.cmp_temporal(tid_center,goes_b5_tid_center))
print('m2 ~noon: ',hex(tid_interval[12]),tid_intervals[12])
print('m2 ~noon: ',hex(gd.stare_set_temporal_resolution(tid_interval[12],gd.stare_temporal_resolutions[2]['1/2day'])))


print('m2 ~noon: ',int(tid_interval[12]/4)&63,tid_intervals[12])
print('m2 ~noon: ',int(gd.stare_set_temporal_resolution(tid_interval[12],gd.stare_temporal_resolutions[2]['1/2day'])/4)&63)
print('m2 ds:    ',hex(gd.merra2_stare_time_ds(ds)),gd.datetime_from_stare(gd.merra2_stare_time_ds(ds)))

print('cmp: m2 noon, g5 tid: ',ps.cmp_temporal([gd.stare_set_temporal_resolution(tid_interval[12],gd.stare_temporal_resolutions[2]['1/2day'])],goes_b5_tid_center))
 
