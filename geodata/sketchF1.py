
import geodata as gd
from join_goes_merra2 import join_goes_and_m2_to_h5


for i in range(3,6):
    print("/home/mrilee/data/goes10.2005.349.003015.BAND_0%d.nc"%i)

data_dir="/home/mrilee/data/"
data_filenames=[]
times=["010015","013015","020015","023015"]
bands=[3,4,5]
pattern="goes10.2005.349.%s.BAND_0%d.nc"
for t in times:
    l = []
    for b in bands:
        l.append(pattern%(t,b))
    data_filenames.append(l)

# print('df: ',data_filenames)

m2_data_dir='/home/mrilee/data/'
m2_filename='MERRA2_300.tavg1_2d_slv_Nx.20051215.nc4'



for filenames in data_filenames:
    tid = gd.temporal_id_from_file(data_dir,filenames[0])
    tid_str = gd.hex16(tid)
    print('working on ','sketchF1.%s.h5'%tid_str,' for ',filenames)
    join_goes_and_m2_to_h5(data_dir,filenames,m2_data_dir,m2_filename,'sketchF1.%s.h5'%tid_str)
    print('---\n')
