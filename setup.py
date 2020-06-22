#!/usr/bin/env/python
"""Installation script
   Copyright (C) 2020 Rilee Systems Technologies LLC
"""

import os
import numpy
from setuptools import setup, Extension, find_packages
from setuptools.command.build_py import build_py as _build_py

LONG_DESCRIPTION = """ """
INSTALL_REQUIRES = ['pystare>=0.5']

# get all data dirs in the datasets module
data_files = []

setup(
    name='geodata',
    version='0.0.2',
    packages=find_packages(),
    description="",
    long_description=LONG_DESCRIPTION,         
    py_modules = ['geodata'],
    python_requires=">=3.5",
    install_requires=INSTALL_REQUIRES
) 

