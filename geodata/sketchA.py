
# Check terminators at various resolutions.

import geodata as gd
import numpy as np
import pystare as ps

# sid    = 1
# sid      = 0x7000000000000000+27
sid    = 0x799d1bcabd6f9260+11
# sid    = 0x799d1bcabd6f9260+0
# sid    = 0x799d1bcabd6f0000 + 27
# sid    = 0x799d1bcabd6f0000 + 26
# sid    = 0x009d1bcabd6f0000 + 27
# sid      = 0x7fffffffffffffd8
print('resolution ',gd.spatial_resolution (sid))
print('latlon     ',ps.to_latlon([sid]))

resolution = gd.spatial_resolution (sid)
mask = gd.spatial_terminator_mask(gd.spatial_resolution(sid))
print('sid  x%016x'%sid)
print('sid0 x%016x'%((sid & ~mask)+resolution))
print('mask x%016x'%mask)
sid_term = sid | ( gd.spatial_terminator_mask(gd.spatial_resolution(sid)))
print('term x%016x'%sid_term)
# sid_term = sid_term - 31 + 27
# print(ps.to_latlon([sid_term]))

exit()

print(hex(sid))
print(hex(sid_term))
print(sid & 31)
print(sid_term & 31)

# sid = 0x0 + (1 << 59) + 0
sid = 0x0 + 7*(1 << 59) + 0
# sid = 0x1000000000000000
# sid = 0x7000000000000000 >> 1
print(0,'--   x%016x'%sid)
for i in range(28):
    print('-')
    sida = np.array([sid+i],dtype=np.int64)
    sidb = sida | gd.spatial_terminator_mask(gd.spatial_resolution(sida[0]))
    # print('sida ',hex(sida[0]))
    # print('sida x%016x'%sida[0])
    # print('mask x%016x'%gd.spatial_terminator_mask(gd.spatial_resolution(sida[0])))
    print(i,'sidb x%016x'%sidb[0])
    id,idt=ps.from_intervals(sida)
    # print('--   x%016x'%id[0])
    print(i,'--   x%016x'%idt[0])
    print(i,'-+   x%016x'%gd.spatial_terminator(sida)[0])
    # print('      fedcba9876543210')
