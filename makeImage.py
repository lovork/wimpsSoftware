import astropy
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import aplpy
import numpy as np
import matplotlib
import os

def findFITSFile():
    for file in os.listdir():
        if file.endswith('.fits'):
            out = file
    try:
        return out
    except UnboundLocalError:
        print('No FITS file found!')

def findRegionFile():
    for file in os.listdir():
        if file.endswith('.fits'):
            out = file
    try:
        return out
    except UnboundLocalError:
        print('No region file found!')


frequency = '143.651 MHz'
title = r'Canes Venatici I without fakesource, $\nu = $'+frequency
imagename = 'CVNI_nosource.png'

gc = aplpy.FITSFigure(data=findFITSFile())

gc.set_title(title)
try:
    gc.show_grayscale(vmin=-0.00227744, vmax=0.00526248)
    gc.add_colorbar()
except:
    print('Unexpected error in the loading process of the FITS file.')


try:
    gc.show_regions(region_file=str(findRegionFile()), layer='region')
    gc.show_layer('region')
except:
    print('Unexpected error in the loading process of the region file.')

gc.savefig(imagename)
