#!/usr/bin/env python

import pystare
import numpy
import matplotlib.pyplot as plt
# from pyhdf.SD import SD

lat = numpy.array([30, 45, 60], dtype=numpy.double)
lon = numpy.array([45, 60, 10], dtype=numpy.double)

indices = pystare.from_latlon(lat, lon, 14)
intersection=pystare.intersect(indices,indices,multiresolution=False)
print('indices:      0x%016x 0x%016x 0x%016x'%tuple(indices))
print('intersection: 0x%016x 0x%016x 0x%016x'%tuple(intersection))


