#this script manipulates a pre existing fits-mask in ds9-format by adding a circular shaped region.
#the region must be in ds9-.reg format.
#the .mask-file and the .reg-file will be given as parameters.
#@author: Finn Welzmüller, Universität Hamburg

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.wcs.utils as au
from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import FK5  
import astropy.units as u
import sys

#extract header information and data from a primary HDU. data should have (1,1,x,y)-shape!
#@param name: The name of the fits file
#@return: header and data of the fits file
def loadFitsFile(name):
    fitsname=get_pkg_data_filename(name)
    f=fits.open(fitsname)
    if len(f[0].data.shape) != 4:
        raise AttributeError('data shape doesn\'t fit! (Must be (1,1,x,y))')
    return f[0].header, f[0].data


#loads the world coordinate system from a fits file or mask
#@param data_header: the header of the fits file or mask
#(must be loaded with loadFitsFile before)
#@return: the WCS
def loadWCS(data_header):
    return WCS(data_header)


#returns a length in pixels when a length in arcsecs is given
#@param arcsecs: length in arcsecs
#@param header: header from the fits file or mask
#@return: length in pixels
def arcsecsToPixels(arcsecs,header):
	return arcsecs/(header['CDELT2']*3600)

#returns a length in arcsecs when a length in pixels is given
#@param pixels: length in pixels
#@param header: header from the fits file or mask
def pixelsToArcsecs(pixels,header):
	return pixels*header['CDELT2']*3600


#loads a ds9 region file and extracts information. Works only for circular regions
#@param regionname: name of the region file
#@param data_header: header of a mask file 
#@return: region information (region center and radius) in pixels
def loadRegionFile(regionname, data_header):
    r = open(regionname, 'r')
    for line in r.readlines():
        if 'circle' in line:
            region = line[line.find('(')+1 : line.find(')')].split(',')
    c = SkyCoord(str(region[0])+" "+str(region[1]), frame=FK5, unit=(u.hourangle, u.deg))
    if region[2].endswith('"'):
	    radius = float(region[2].replace('"', ''))
    elif region[2].endswith("'"):
	    radius = float(region[2].replace("'", ''))*60
    else:
        radius = float(region[2])*3600
    
    WCS = loadWCS(data_header)
    return au.skycoord_to_pixel(c, WCS), arcsecsToPixels(radius, data_header)

	
#stances a filled circle on a given position and a given radius in a .mask-file
#@param inp_array: input mask array (loaded from a .mask-file)
#@param xcenter, ycenter: central pixel from the circular region
#@param radius: radius of the circular region
#@return: array with the additional region

def stanceMask(inp_array, xcenter, ycenter, radius):
    inp_array = np.array(inp_array)
    for x in range(0, inp_array.shape[0]-1,1):
        for y in range(0, inp_array.shape[1]-1,1):
            if (x-xcenter)**2 + (y-ycenter)**2 <= radius**2:
                inp_array[x][y] = 1
    return inp_array    

#main method
#@param maskName: Name of the pre existing mask
#@param regionName: Name of the ds9 region file (must be of CIRCULAR shape!)
def main(maskName, regionName):
    mask_header, mask_data = loadFitsFile(maskName)
    center, radius = loadRegionFile(regionName, mask_header)
    newMask = stanceMask(inp_array=mask_data[0][0][:][:], xcenter=center[0], ycenter=center[1], radius=radius)
    thirdAxis = []
    thirdAxis.append(newMask)
    fourthAxis = []
    fourthAxis.append(thirdAxis)

    hdu = fits.PrimaryHDU(fourthAxis)
    hdul = fits.HDUList([hdu])
    hdul.writeto('newMask.mask.fits', overwrite=True)

if __name__ =='__main__':
    main(str(sys.argv[1]), str(sys.argv[2]))


