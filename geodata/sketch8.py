
import datetime as dt
import geodata as gd
import numpy as np
import pystare as ps

gfname       = "goes10.2005.349.223015.BAND_03.nc"
def temporal_id_centered_from_goes_filename(gfname):
    gfname_split = gfname.split(".")
    yr  = gfname_split[1]
    ydy = gfname_split[2]
    hr  = int(gfname_split[3][0:2])
    mn  = int(gfname_split[3][2:4])
    sec = int(gfname_split[3][4:6])
    gdt = dt.datetime(int(yr),1,1)+dt.timedelta(int(ydy)-1)
    gdt_str = gd.format_time(int(yr),int(gdt.month),int(gdt.day),hr,mn,sec)
    gdt_np = np.array([gdt_str],dtype='datetime64[ms]')
    gtid_centered = ps.from_utc(gdt_np.astype(np.int64),gd.stare_temporal_resolutions[2]['1/4hr'])
    return gtid_centered

gtid_centered = temporal_id_centered_from_goes_filename(gfname)

#   print('g ',gfname_split)
#   print('g ',hr,mn,sec)
#   print('g ',gdt)
#   print('g ',gdt.year)
#   print('g ',gdt.month)
#   print('g ',gdt.day)
#   print('g ',gdt_str)
#   print('gdt_np',gdt_np,hex(gtid_centered[0]))

m2name       = "MERRA2_300.tavg1_2d_slv_Nx.20051215.nc4"

def temporal_id_centered_from_merra2_filename(m2name):
    m2name_split = m2name.split(".")
    yr  = int(m2name_split[2][0:4])
    mo  = int(m2name_split[2][4:6])
    dy  = int(m2name_split[2][6:8])
    hr  = 12
    mn  = 0
    sec = 0
    m2dt_str = gd.format_time(yr,mo,dy,hr,mn,sec)
    m2dt_np = np.array([m2dt_str],dtype='datetime64[ms]')
    m2tid_centered = ps.from_utc(m2dt_np.astype(np.int64),gd.stare_temporal_resolutions[2]['1/2day'])
    return m2tid_centered

m2tid_centered = temporal_id_centered_from_merra2_filename(m2name)
#  print('m2',m2name_split)
#  print('m2',m2dt_str)
#  print('m2dt_np',m2dt_np,hex(m2tid_centered[0]))

print('')
print('cmp: g,m2: ',ps.cmp_temporal(gtid_centered,m2tid_centered))

print('')
print('gd version')
print('g,m2:      ',hex(gd.temporal_id_centered_from_filename(gfname)),hex(gd.temporal_id_centered_from_filename(m2name)))
print('cmp: g,m2: ',ps.cmp_temporal([gd.temporal_id_centered_from_filename(gfname)],[gd.temporal_id_centered_from_filename(m2name)]))


