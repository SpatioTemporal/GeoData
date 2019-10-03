
# Look for intersection between  MERRA and GOES files
# Ignoring the file-name connection...

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


# HDF vs. ADM... Nice way to mock...

m2_pattern="MERRA*.nc4"
goes_pattern="goes*.nc"


m2_tid_index = {}
for entry in listOfFiles:
    if fnmatch.fnmatch(entry,m2_pattern):
        tid = gd.temporal_id_from_file(dataPath,entry)
        if tid not in m2_tid_index.keys():
            m2_tid_index[tid] = [entry]
        else:
            m2_tid_index[tid].append(entry)

print('m2 tid keys:', list(m2_tid_index.keys()))
        
for entry in listOfFiles:
    if fnmatch.fnmatch(entry,goes_pattern):
        gtid = gd.temporal_id_from_file(dataPath,entry)
        # print(entry,gtid)
        # print('gtid: ',hex(gtid),gd.datetime_from_stare(gtid))
        gm2_match = ps.cmp_temporal(np.array([gtid],dtype=np.int64),list(m2_tid_index.keys()))
        match_fnames = []
        for i in  range(gm2_match.size):
            if gm2_match[i] == 1:
                fine_match = ps.cmp_temporal(np.array([gtid],dtype=np.int64),gd.merra2_stare_time(Dataset(dataPath+m2_tid_index[list(m2_tid_index.keys())[i]][0])))
                # print('fine_match: ',fine_match)
                #print('m2tid: ',gd.datetime_from_stare(list(m2_tid_index.keys())[i]))
                if 1 in fine_match:
                    match_fnames.append(m2_tid_index[list(m2_tid_index.keys())[i]][0])
                else:
                    match_fnames.append(None)
            else:
                match_fnames.append(None)
        # print(entry, ' entry,matches: ',gm2_match,match_fnames)
        match_fnames_trimmed = []
        for i in match_fnames:
            if i is not None:
                match_fnames_trimmed.append(i)
        # print(entry,match_fnames_trimmed)
        if(len(match_fnames_trimmed) > 1):
            print('*** WARNING: more than one MERRA-2 file for the GOES file!!')
        matched_pair = (entry,match_fnames_trimmed[0])
        print('matched_pair: ',matched_pair)

