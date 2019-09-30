
# Load MERRA-2 times and convert to STARE.

from netCDF4 import Dataset
import numpy as np
import pystare as ps

from geodata import merra2_parse_begin_date, merra2_make_time, format_time

dataPath   = "/home/mrilee/data/"
dataFile   = "MERRA2_300.tavg1_2d_slv_Nx.20051215.nc4"
fqFilename = dataPath+dataFile
ds         = Dataset(fqFilename)

# print('keys: ',ds.variables.keys())



tim = ds['time']
print('tim:                  ',tim)
print('tim units:            ',tim.units)
print('tim begin data:       ',tim.begin_date)
print('tim m2 prs date:      ',merra2_parse_begin_date(tim.begin_date))
print('tim begin data typ:   ',type(tim.begin_date))
print('tim begin time:       ',tim.begin_time)
# print('tim[:]: ',tim[:])

# Resolution level
temporal_format_type   =  2
resolution_level_1day  = 27
resolution_level_1hr   = 32
resolution_level_1o2hr = 33
resolution_level_1o4hr = 34

print('')
for i in range(len(tim)):
    print('%3d i,tim %04d '%(i,tim[i]),' %02d:%02d '%merra2_make_time(tim.begin_time/100,tim[i]))
# Note: we could use the preceding to make temporal indexes with a resolution level of 1/2 hr
# Since the cmp routine calculates a difference and is not necessarily using integer intervals.

# For intervals we could use the following with a different scheme for setting the index.
# This adds the silliness of how to interpret our intervals. If we keep the STARE scheme
# The start of the interval is the start, but then cmp-at-resolution is not quite right.
# Note cmp does the right thing.
print('')
for i in range(len(tim)):
    print('%3d i,tim %04d '%(i,tim[i]),' %02d:%02d '%merra2_make_time(0,tim[i]))

formatted_times = []
idx = np.zeros([tim.size],dtype=np.int64)
for i in range(len(tim)):
    yr,mo,dy = merra2_parse_begin_date(tim.begin_date)
    hr,mn    = merra2_make_time(0,tim[i])
    sc       = 0
    tm       = format_time(yr,mo,dy,hr,mn,sc)
    dt = np.array([tm],dtype='datetime64[ms]')
    idx[i]   = ps.from_utc(dt.astype(np.int64),resolution_level_1hr)
    # idx[i]   = ps.from_utc(dt.astype(np.int64),resolution_level_1o4hr)
    back     = np.array(ps.to_utc_approximate(np.array([idx[i]],dtype=np.int64)),dtype='datetime64[ms]')
    print(yr,mo,dy,hr,mn,tm,dt,hex(idx[i]),back)
    formatted_times.append(tm)

### GOES BAND_05 # Check time information

goes_b5_dataPath = "/home/mrilee/data/"
goes_b5_dataFile = "goes10.2005.349.003015.BAND_05.nc"
goes_b5_fqFilename = goes_b5_dataPath+goes_b5_dataFile
goes_b5_ds = Dataset(goes_b5_fqFilename)

# print('keys: ',goes_b5_ds.variables.keys())

tim = goes_b5_ds['time']
print('tim:                  ',tim)
print('tim units:            ',tim.units)
print('tim[0]:               ',tim[0])
print('tim[0].shape:         ',tim[0].shape)
print('type(tim[0]):         ',type(tim[0]))
dt = np.array(tim[0]*1000,dtype='datetime64[ms]')
dt = dt.reshape([1])
print('type(dt): ',type(dt))
print('dt.size:  ',dt.size)
print('dt.shape: ',dt.shape)
print('dt:       ',dt)

gb5idx = ps.from_utc(dt.astype(np.int64),resolution_level_1o4hr)
print('gb5idx shape,size:  ',gb5idx.shape,gb5idx.size)
print('idx shape,size:     ',idx.shape,idx.size)

gb5idx_back = np.array(ps.to_utc_approximate(np.array([gb5idx[0]],dtype=np.int64)),dtype='datetime64[ms]')

print('dt,gb5idx,back: ',dt[0],hex(gb5idx[0]),gb5idx_back)


# for i in range(idx.size):
#     print(i,' i,gb,idx : ',gb5idx[0],idx[i])

# for i in range(idx.size):
#     print(i,' i,gb,idx : ',hex(gb5idx[0]),hex(idx[i]),hex(gb5idx[0] & ~idx[i]))

# print('idx vs. idx:    ',ps.cmp_temporal(idx,idx))
print('cmp g5 vs. idx:    ',ps.cmp_temporal(gb5idx,idx))
print('cmp idx vs. g5:    ',ps.cmp_temporal(idx,gb5idx))
print('cmp idx[0] vs. g5: ',ps.cmp_temporal(np.full([gb5idx.size],idx[0:1],dtype=np.int64),gb5idx))

back_utc = ps.to_utc_approximate(gb5idx)
print('back check: ',back_utc)
print('back check: ',np.array(back_utc,dtype='datetime64[ms]'))


