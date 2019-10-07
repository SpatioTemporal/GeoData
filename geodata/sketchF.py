#!/usr/bin/env python3

import geodata as gd
import os,fnmatch
import yaml

with open("config.yaml") as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
    print(config)
    print('')
    if config['run']['action'] == "list":
        print("%s %s"%("list",config['data_source']['directory']))

class data_catalog(object):
    def __init__(self,config):
        self.config = config
        self.files  = None
        return

    def get_files(self):
        if self.files is None:
            self.files = []
            dir      = self.config['directory']
            filelist = os.listdir(dir)
            patterns = self.config['patterns']
            for pattern in patterns:
                for entry in filelist:
                    if fnmatch.fnmatch(entry,pattern):
                        self.files.append(entry)        
        return self.files

    def get_tid_centered_index(self):
        return gd.temporal_id_centered_filename_index(self.get_files())

goes_b5_catalog = data_catalog(config['data_sources']['goes_gvar_img_b5'])
goes_files = goes_b5_catalog.get_files()

# goes_files = []
# cfg = config['data_sources']['goes_gvar_img_b5']
# dir      = cfg['directory']
# files=os.listdir(dir)
# patterns = cfg['patterns']
# for pattern in patterns:
#     for entry in files:
#         if fnmatch.fnmatch(entry,pattern):
#             goes_files.append(entry)
#             # print(entry)


m2_catalog = data_catalog(config['data_sources']['merra2'])
m2_files = m2_catalog.get_files()

# m2_files = []
# cfg = config['data_sources']['merra2']
# dir      = cfg['directory']
# files=os.listdir(dir)
# patterns = cfg['patterns']
# for pattern in patterns:
#     for entry in files:
#         if fnmatch.fnmatch(entry,pattern):
#             m2_files.append(entry)
#             # print(entry)

m2_tid_index = m2_catalog.get_tid_centered_index()

dir = m2_catalog.config['directory']

for entry in goes_files:
    gtid = gd.temporal_id_centered_from_filename(entry)
    print('matched pair: 0x%016x % 40s % 40s'%(gtid,entry,gd.temporal_match_to_merra2(gtid,m2_tid_index,dataPath=dir)[0]))   
