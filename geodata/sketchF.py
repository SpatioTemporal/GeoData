#!/usr/bin/env python3

import os,fnmatch
import yaml

with open("config.yaml") as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
    print(config)
    print('')
    if config['run']['action'] == "list":
        print("%s %s"%("list",config['data_source']['directory']))

cfg = config['data_sources']['goes_gvar_img']

dir      = cfg['directory']
patterns = cfg['patterns']
files=os.listdir(dir)
for pattern in patterns:
    for entry in files:
        if fnmatch.fnmatch(entry,pattern):
            print(entry)

cfg = config['data_sources']['merra2']

dir      = cfg['directory']
patterns = cfg['patterns']
files=os.listdir(dir)
for pattern in patterns:
    for entry in files:
        if fnmatch.fnmatch(entry,pattern):
            print(entry)



