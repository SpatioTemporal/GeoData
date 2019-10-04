
from dask.distributed import Client
import numpy as np
import pystare as ps

from time import process_time as timer

def hex16(i):
    return "0x%016x"%i

def main():
    
    # client = Client(processes = False) # threads ?
    client     = Client()
    size       = 10000000
    # size       = 20
    # shards     = 20
    # shards     = 6
    # shards     = 1
    shards     = 12
    shape      = [size]
    lat        = np.random.rand(size)*180.0-90.0
    lon        = np.random.rand(size)*360.0-180.0
    resolution_ = 8
    resolution = np.full(shape,resolution_,dtype=np.int64)

    # print('lat shape: ',lat.shape)

    print('')
    serial_start = timer()
    s_sids = ps.from_latlon(lat,lon,resolution_)
    s_sidsstr = [hex16(s_sids[i]) for i in range(len(s_sids))]
    serial_end = timer()
    # print('0 s_sids: ',s_sids)
    print('time s_sids: ',serial_end-serial_start)

    def w_from_latlon(llr):
        # print('')
        # print('llr:  ',llr)
        sids = ps.from_latlon(llr[0],llr[1],int(llr[2][0]))
        # print('sids: ',sids)
        # print('')
        return sids
    
    # def w_from_latlon1(lat,lon,res):
    #     return ps.from_latlon(np.array([lat],dtype=np.double)\
    #                            ,np.array([lon],dtype=np.double)\
    #                            ,int(res))
    # sid        = ps.from_latlon(lat,lon,resolution)
    # sid        = client.map(w_from_latlon1,lat,lon,resolution) # futures

    dask_start = timer()
    shard_size = int(size/shards)
    shard_bins = np.arange(shards+1)*shard_size
    shard_bins[-1] = size

    # print('---')
    # print('shards:     ',shards)
    # print('shard_size: ',shard_size)
    # print('shard_bins: ',shard_bins)
    # print('---')
    lat_shards = [lat[shard_bins[i]:shard_bins[i+1]] for i in range(shards)]
    lon_shards = [lon[shard_bins[i]:shard_bins[i+1]] for i in range(shards)]
    res_shards = [resolution[shard_bins[i]:shard_bins[i+1]] for i in range(shards)]
    
    llr_shards = []
    for i in range(shards):
        llr_shards.append([lat_shards[i],lon_shards[i],res_shards[i]])

    # print('llr_shards len: ',len(llr_shards))
    # print('llr_shards: ',llr_shards)

    ## future = client.submit(func, big_data)    # bad
    ## 
    ## big_future = client.scatter(big_data)     # good
    ## future = client.submit(func, big_future)  # good

    # sid        = client.map(w_from_latlon,llr_shards) # futures

    big_future = client.scatter(llr_shards)
    sid        = client.map(w_from_latlon,big_future) # futures

    # print('0 sid:  ',sid)
    # print('9 len(sid): ',len(sid))
    # for i in range(shards):
    #     print(i, ' 10 sid: ',sid[i])
    #     print(i, ' 11 sid: ',sid[i].result())

    # print('15 sid:    ',[type(i) for i in sid])

    sid_cat = np.concatenate([i.result() for i in sid])
    sidsstr = [hex16(sid_cat[i]) for i in range(len(sid_cat))]
    dask_end = timer()
    # print('2 sids: ',sids)
    sids = sid_cat
 
    print('')
    # for i in range(size-20,size):
    for i in np.array(np.random.rand(20)*size,dtype=np.int64):
        print("%09i"%i,sidsstr[i],s_sidsstr[i],' ',sids[i]-s_sids[i])

    print('')
    print('dask total threads:  ',sum(client.nthreads().values()))
    print('size:                ',size)
    print('shards:              ',shards)
    print('')
    print('time sids:           ',dask_end-dask_start)
    print('time s_sids:         ',serial_end-serial_start)
    print('parallel speed up:   ',(serial_end-serial_start)/(dask_end-dask_start))

    client.close()

if __name__ == "__main__":
    main()


# countour at 2cm
