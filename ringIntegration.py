#main script that produces radial profiles and FWHM plots.
#The script uses PyBDSF, WSClean and NDPPP. Make sure its avilable. 
#the script needs a information file with name 'ringIntOpts.txt'. 
#Use the clean version that is avilable in the repo and remove the '_clean'
#after storing the information.
#make sure that there is only one ds9 region file in the folder you run the script in
#that has circular shape. In this region, the radial profile will be created.
#as output, the script will create three folders: 
#in 'nosources.../', the output from WSClean without fakesource will be stored.
#in 'fakesource.../', the output from WSclean with fakesource will be stored.
#in 'output/', all plots will be stored (in png and txt format).
#@author: Finn Welzmüller, Universität Hamburg


#!/usr/bin/python3
import matplotlib
matplotlib.use('Agg')
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.wcs.utils as au
from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
from scipy.optimize import curve_fit
import os
from shutil import copyfile
import bdsf
import json
import astropy.convolution as co
#from scipy.signal import convolve as scipy_convolve
from  astropy.convolution import convolve as scipy_convolve

#loads values from the 'ringIntOpts.txt' option sheet.
#make sure to remove the '_clean' from the file name if you are using a new version!
#@return: all values stored in the option sheet.
def loadOptions():
	r = open("ringIntOpts.txt", "r")
	lines = r.readlines()
	for line in lines:
		if line.startswith("fitsFileName"):
			fitsFileName = line[15:len(line)-1]
		if line.startswith("fakeSourceArray"):
			fakeSourceArray = line[18:len(line)-1]
		if line.startswith("uvCutArray"):
			uvCutArray = line[13:len(line)-1]
		if line.startswith("robustArray"):
			robustArray = line[14:len(line)-1]
		if line.startswith("taperArray"):
			taperArray = line[13:len(line)-1]
		if line.startswith("wscleanSize"):
			wscleanSize = line[14:len(line)-1]
		if line.startswith("baselineParameter"):
			baselineParameter = line[20:len(line)-1]
		if line.startswith("beamInformation"):
			beamInformation = line[18:len(line)-1]
		if line.startswith("FWHMmin"):
			FWHMmin = line[10:len(line)-1]
		if line.startswith("FWHMmax"):
			FWHMmax = line[10:len(line)-1]
		if line.startswith("FWHMstep"):
			FWHMstep = line[11:len(line)-1]
		if line.startswith("inputDirectoryArray"):
			inputDirectoryArray = line[21:len(line)-1]
		if line.startswith("wscleanScale"):
			wscleanScale = line[15:len(line)-1]
		if line.startswith("galFWHM"):
			galFWHM = line[10:len(line)-1]
		if line.startswith("folding_do"):
			folding_do = line[13:len(line)-1]
		if line.startswith("targetbmaj"):
			targetbmaj = line[13:len(line)-1]
		if line.startswith("targetbmin"):
			targetbmin = line[13:len(line)-1]
		if line.startswith("targetTheta"):
			targetTheta = line[14:len(line)-1]
	    
	print("fitsfileName: "+str(fitsFileName))
	print("fakeSouceArray: "+str(fakeSourceArray))
	print("uvCutArray: "+str(uvCutArray))
	print("robustArray: "+str(robustArray))
	print("taperArray: "+str(taperArray))
	print("wscleanSize: "+str(wscleanSize))
	print("baselineParameter: "+str(baselineParameter))
	print("beamInformation: "+str(beamInformation))
	print("FWHMmin: "+str(FWHMmin))
	print("FWHMmax: "+str(FWHMmax))
	print("FWHMstep: "+str(FWHMstep))
	print("inputDirectoryArray: "+str(inputDirectoryArray))
	print("wscleanScale: "+str(wscleanScale))
	print("galFWHM: "+str(galFWHM))
	print("folding_do: "+str(folding_do))
	print("targetbmaj: "+str(targetbmaj))
	print("targetbmin: "+str(targetbmin))
	print("targetTheta: "+str(targetTheta))
    
	return fitsFileName, fakeSourceArray, uvCutArray, robustArray, taperArray, wscleanSize, baselineParameter, beamInformation, FWHMmin, FWHMmax, FWHMstep, inputDirectoryArray, wscleanScale, galFWHM, folding_do, targetbmaj, targetbmin, targetTheta

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
    

#returns the mean and standard deviation from a circular disk on the map
#@param xc, yc: center PIXELS from the circular disk
#@param rmin, rmax: radii of the circular disk
#@param data_arr: fits file data_array
#@param header: header from the primary HDU
#@param: beamInformation: x and y extension of the beam
#@return: flux mean and flux standard deviation (per beam, not per pixels)
def ringIntWithMean(xc, yc, rmin, rmax, data_arr, header, beamInformation):
	i = xc - rmax - 5
	j = yc - rmax - 5
	flux_array = []
	px = 0

	for i in range(int(xc - rmax - 5), int(xc + rmax + 5)):
		for j in range(int(yc - rmax - 5), int(yc + rmax + 5)):
			if (rmin**2 < ((i - xc)**2 + (j - yc)**2)) and (((i - xc)**2 + (j - yc)**2) <= rmax**2):
				flux_array.append(data_arr[i][j])
				px +=1    
	
	#20 = folded FWHM
	n_beams = np.pi*(pixelsToArcsecs(rmax, header)**2 - pixelsToArcsecs(rmin, header)**2)/(1.133*20**2)
	return np.mean(flux_array), np.std(flux_array)/np.sqrt(n_beams)


#same as ringIntWithMean(...) but for beam construction.
def ringIntWithMeanPSF(xc, yc, rmin, rmax, data_arr, header, beamInformation):
	i = xc - rmax - 5
	j = yc - rmax - 5
	flux_array = []
	px = 0

	for i in range(int(xc - rmax - 5), int(xc + rmax + 5)):
		for j in range(int(yc - rmax - 5), int(yc + rmax + 5)):
			if (rmin**2 < ((i - xc)**2 + (j - yc)**2)) and (((i - xc)**2 + (j - yc)**2) <= rmax**2):
				flux_array.append(data_arr[0][0][i][j])
				px +=1    
	
    #20 = folded FWHM
	n_beams = np.pi*(pixelsToArcsecs(rmax, header)**2 - pixelsToArcsecs(rmin, header)**2)/(1.133*20**2)
	return np.mean(flux_array), np.std(flux_array)/np.sqrt(n_beams)

#returns a radial flux profile of a circle shaped area with a radius and it's center in PIXELS (xc,yc). 
#The flux will be averaged over a circular disk with radius steps. The distance to the center will be returned
#in the array distance_arr, the averaged flux will be returned in the array flux_arr.
#@param xc, yc: central coodinates of the region
#@param steps: the amount the radius will grow when finding the next ring
#@param radius: maximal radius for the radial profile
#@param data_arr: data in which the the radial profile should be calculated
#@return distance_arr: array with the distances to the center or the radii from the circles
#@return flux_arr: array with the averaged flux in each ring
#@return yerror: array with the standard deviations of the flux in each ring
#@return fluxsum: total summed flux (has errors because of the averaging)
#@return pix_arr: amount of pixels considered in the calculation
def findRadialProfile(xc, yc, steps, radius, data_arr, header, beamInformation):
	steps = arcsecsToPixels(steps,header)
	distance_arr = []
	yerror = []
	flux_arr = []
	r = 0

    
	while r <= radius:
	
		fluxmean, fluxstd = ringIntWithMean(xc=xc, yc=yc, rmin=r, rmax=r+steps, data_arr=data_arr,header=header, beamInformation=beamInformation)
		
		distance_arr.append(pixelsToArcsecs(r,header))
		flux_arr.append(fluxmean)
		yerror.append(fluxstd)
		r += steps
	return distance_arr, flux_arr, yerror


#same as findRadialProfile(...) but for beam construction
def findRadialProfilePSF(xc, yc, steps, radius, data_arr, header, beamInformation):
	steps = arcsecsToPixels(steps,header)
	distance_arr = []
	yerror = []
	flux_arr = []
	r = 0

    
	while r <= radius:
	
		fluxmean, fluxstd = ringIntWithMeanPSF(xc=xc, yc=yc, rmin=r, rmax=r+steps, data_arr=data_arr,header=header, beamInformation=beamInformation)
		
		distance_arr.append(pixelsToArcsecs(r,header))
		flux_arr.append(fluxmean)
		yerror.append(fluxstd)
		r += steps
	return distance_arr, flux_arr, yerror


#loads a ds9 regions file and converts region from degree units to pixels
#@param region: region file in ds9 format (must contain a circular shape!)
#@param wcs: world coordinate system stored in the overall fits file
#@param data_header: HDU header of the overall fits file
#@return: array that contains the region information in PIXEL format. 
#(region_array[0]: center pixel in x-direction, region_array[1]: center pixel in
#y-direction, region_array[2]: region radius)
def loadRegionFile(region, wcs, data_header):
	print("region in loadRegionFile: "+str(region))
	region_array = []
	print(region[0])
	print(region[1])
	c= SkyCoord(str(region[0])+" "+str(region[1]), frame=FK5, unit=(u.hourangle, u.deg))
	if region[2].endswith('"'):
		radius = float(region[2].replace('"', ''))/3600
	elif region[2].endswith("'"):
		radius = float(region[2].replace("'", ""))/60
	print("radius after loadRegionFile: "+str(radius))
	reg_center = au.skycoord_to_pixel(c, wcs=wcs)
	region_array.append(reg_center[0])
	region_array.append(reg_center[1])
	region_array.append(arcsecsToPixels(3600*float(radius), header=data_header))
	return region_array


#writes any data to a txt file
#@param path: output path of the file
#@param name: name of the txt file
#@param xdata: data of the abcissa
#@param ydata: data of the ordinate
#@param yerror: error on ydata
def writeData(path, name, xdata, ydata, yerror):
	with open(str(path)+str(name)+".txt","w") as n:
		l=0 
		while l<=len(xdata)-1:
			n.write(str(xdata[l])+"\t"+str(ydata[l])+"\t"+str(yerror[l])+"\n")
			l+=1

#simple gaussian with an amplitude a and a FWHM binit. Needed for fitting
def gaussian(x,a):
	return a*np.e**((-x**2)/(2*binit**2))


#creates a profile of fitted gaussian amplitudes for different FWHMs.
#@param FWHMmin: FWHM of the narrowest Gaussian
#@param FWHMmax: FWHM of the wides Gaussian 
#@param steps: step size between two fitting FWHMs
#@param xdata: abcissa data for fitting
#@param ydata: ordinate data for fitting
#@param yerror: error on ordinate data for fitting
def findFitProfile(FWHMmin,FWHMmax,steps,xdata,ydata,yerror):
    
	def gaussian(x,a):
		return a*np.e**((-x**2)/(2*b**2))

	width = []
	amplitude = []
	error = []
    
	bmin=FWHMmin/2.355
	bmax=FWHMmax/2.355
    

	while bmin <=bmax:
		b=bmin
		width.append(b*2.355)
		popt,pcov=curve_fit(f=gaussian,xdata=xdata,ydata=ydata,sigma=yerror)
		amplitude.append(popt[0])
		error.append(np.sqrt(pcov[0][0]))
		bmin+=(steps/2.355)
	return width,amplitude,error

#scales an array with a factor (basically useless in python3). 
#Can be used eg for scaling from arcsec to arcmin for from Jy/beam to microJy/beam
#@param inp: input array
#@param factor: scaling factor
#@return scaled array
def rescaleArray(inp,factor):
	out=[]
	i=0
	while i<len(inp):
		out.append(factor*inp[i])
		i+=1
	return out

#indentifying point like sources using PyBDSF 
#Usually I do this step by myself in PyBDSF. 
#An already existing point source catalog can be written into the main method.
#@param fitsFileName: the.fits-file PyBDSF works with
#@return the name of the Catalog with identified point sources in .bbs format
def findPointSources(fitsFileName):
    	#define catalog name (fitsFileName_pointSourceCatalog)
	pointSourceCatalogName = fitsFileName[:len(fitsFileName) - 5] + "_pointSourceCatalog"
	pointSourceCatalog = pointSourceCatalogName + ".bbs"
    
	#perform pyBDSF point source search
	img=bdsf.process_image(fitsFileName,atrous_do=True, rms_box=(60,20))
    
	#save catalog file as .bbs file
	img.write_catalog(outfile=pointSourceCatalog,catalog_type='gaul',clobber=True,format='bbs')
    
	#save catalog file also as .ds9.reg file
	img.write_catalog(outfile=pointSourceCatalogName + ".ds9.reg", catalog_type='gaul', clobber=True, format='ds9')
	print("Point source Catalog: "+str(pointSourceCatalog))
	return pointSourceCatalog

#stores a fake source in the pointSourceCatalog in the center of a regions file with a specified intensity and spreading (FWHM)
#@param pointSourceCatalog: the .bbs file where the fake source will be added in Millijansky
#@param regionsFile: the .reg-file for getting the center of the galaxy
#@param fakeSource_arr: list of fake source information in shape of 
#[[I(fakesource1), FWHM(fakesource1), xpos(fakesource1), ypos(fakesource1)], [I(fakesource2), ...], ...]
#@return: the name of the fake source catalog
def storeFakeSource(pointSourceCatalog, fakeSource):
	#find name for fake source catalog
	fakeSourceCatalog = pointSourceCatalog[:len(pointSourceCatalog) - 23]+"_"+str(fakeSource[0])+"mJy_"+str(fakeSource[1])+"arcsec"+str(fakeSource[2])+","+str(fakeSource[3])+".bbs"
    
	#copying point source catalog
	copyfile(pointSourceCatalog, fakeSourceCatalog)
	r = open(fakeSourceCatalog, "a")
	i = 0
	#writing fake sources...
	r.write("sim_gauss"+str(i)+", GAUSSIAN, "+str(fakeSource[2])+","+str(fakeSource[3])+", -"+str(fakeSource[0])+"e-3,0.,0.,0.,"+str(fakeSource[1])+", "+str(fakeSource[1])+",  0.,1.43651e+08, [-0.8]")
	print("Fake source catalog: "+str(fakeSourceCatalog))
	return fakeSourceCatalog


#subtracting point sources given in a .bbs-catalog-file from data stored in the input directories 
#@param inputDirectory_arr: array that contains the measurement sets
#@param pointSourceCatalog: PSC which will be subtracted
def subtractPointSources(inputDirectory_arr, pointSourceCatalog):
	#change type of pointSourceCatalog:
	print("makesourcedb in="+str(pointSourceCatalog)+" out="+str(pointSourceCatalog)+".sourcedb outtype=blob append=False")
	os.system("makesourcedb in="+str(pointSourceCatalog)+" out="+str(pointSourceCatalog)+".sourcedb outtype=blob append=False")
	for inputDir in inputDirectory_arr:
		os.system("DPPP msin="+str(inputDir)+" msin.datacolumn=DATA msout=. msout.datacolumn=CORRECTED_DATA steps=[predict] predict.type=predict predict.operation=subtract predict.sourcedb="+str(pointSourceCatalog)+".sourcedb")
		
		
		
		

#run WSClean commands for different UV Cuts, weightings and, for negative weightings, with different gaussian tapers.
#the output will be stored in a folder named with the specific combination.
#WSClean will clean images with and without fakesource.
#@param uvCuts_arr: list of lambdas in for the UV cuts.
#@param robust_arr: list of weightings.
#@param taper_arr: list of tapers for negative weightings
#@param fakeSource: fakeSource array loaded from ringIntOpts.txt
#@param wscleanSize: size parameter for WSClean (only for square images)
#@param wscleanBaseLineAv: baseline parameter for averaging in WSClean
#@param inputDirectory_arr: list containing the multiple input directories
#@param pointSourceCatalog: point Source catalog made with PyBDSF
#@param fakeSourceCatalog: point Source catalog with a stored fakesource
#@param wscleanScale: scale arcsec per pixel
#@return array that contains all possible combinations of parameter.
def runWSClean(uvCuts_arr, robust_arr, taper_arr, fakeSource, wscleanSize, wscleanBaseLineAv, inputDirectory_arr, pointSourceCatalog, fakeSourceCatalog, wscleanScale):
	
	combination_arr = []
	####### P R E P A R E   D I R E C T O R I E S #######
	#L-loop
	for wscleanL in uvCuts_arr:
		#getting the name right ("80" -> "080")
		if len(str(wscleanL))<3:
			l_name = "0"+str(wscleanL)
		else:
			l_name = str(wscleanL)
		#R-loop
		for wscleanR in robust_arr:
			#getting the name right ("-0.5" -> m05)
			if wscleanR < 0:
				r_name = "m"+str(((-1)*wscleanR))
			else:
				r_name = str(wscleanR)
			if "." in r_name:
				r_name = str(r_name.replace(".", ""))
			#for negative weightings, looping over tapers (entries in taper_arr)
			if wscleanR < 0:
				#T-loop
				for wscleanT in taper_arr:
					#preparing directories
					if not os.path.exists('UV'+str(l_name)+'_R'+str(r_name)+'_T'+str(wscleanT)):#Directory for fakesource and observation
						os.makedirs('UV'+str(l_name)+'_R'+str(r_name)+'_T'+str(wscleanT))
					combination_arr.append('UV'+str(l_name)+'_R'+str(r_name)+'_T'+str(wscleanT))
					if not os.path.exists('nosources_atrous_UV'+str(l_name)+'_R'+str(r_name)+'_T'+str(wscleanT)):#directory only for observation data
						os.makedirs('nosources_atrous_UV'+str(l_name)+'_R'+str(r_name)+'_T'+str(wscleanT))
					fakeSourceComb = "fakesource"+str(fakeSource[0])+"mJy_"+str(fakeSource[1])+"arcsec_"+str(fakeSource[2])+"_"+str(fakeSource[3])
					if not os.path.exists(fakeSourceComb+'_atrous_UV'+str(l_name)+'_R'+str(r_name)+'_T'+str(wscleanT)):#directory only for fake source data
						os.makedirs(fakeSourceComb+'_atrous_UV'+str(l_name)+'_R'+str(r_name)+'_T'+str(wscleanT))
			else:
				if not os.path.exists('UV'+str(l_name)+'_R'+str(r_name)):#Directory for fakesource and observation
					os.makedirs('UV'+str(l_name)+'_R'+str(r_name))
					combination_arr.append('UV'+str(l_name)+'_R'+str(r_name))
				if not os.path.exists('nosources_atrous_UV'+str(l_name)+'_R'+str(r_name)):#directory only for observation data
					os.makedirs('nosources_atrous_UV'+str(l_name)+'_R'+str(r_name))  
				fakeSourceComb = "fakesource"+str(fakeSource[0])+"mJy_"+str(fakeSource[1])+"arcsec_"+str(fakeSource[2])+"_"+str(fakeSource[3])	
				if not os.path.exists(fakeSourceComb+'_atrous_UV'+str(l_name)+'_R'+str(r_name)):#directory for fakesource data
					os.makedirs(fakeSourceComb+'_atrous_UV'+str(l_name)+'_R'+str(r_name))
	####### W I T O U T   F A K E S O U R C E #######
	#preparation
	subtractPointSources(inputDirectory_arr, pointSourceCatalog)
	#L-loop
	for wscleanL in uvCuts_arr:
		if len(str(wscleanL))<3:
			l_name = "0"+str(wscleanL)
		else:
			l_name = str(wscleanL)
		#R-loop
		for wscleanR in robust_arr:
			if wscleanR < 0:
				r_name = "m"+str((-1)*wscleanR)
			else:
				r_name = str(wscleanR)
			if "." in r_name:
				r_name = str(r_name.replace(".", ""))
			#for negative weightings, looping over tapers (entries in taper_arr)
			if wscleanR < 0:
				#T-loop
				for wscleanT in taper_arr:
					#run actual wsclean with given parameters for observation
					wscleancommand = "wsclean -no-update-model-required -multiscale -beam-size 20 -fits-mask newMask.mask.fits -minuv-l "+str(wscleanL)+".0 -size "+str(wscleanSize)+" "+str(wscleanSize)+" -reorder -weight briggs "+str(wscleanR)+" -weighting-rank-filter 3 -clean-border 1 -mgain 0.8 -fit-beam -data-column CORRECTED_DATA -join-channels -channels-out 6 -padding 1.4 -auto-mask 2.5 -auto-threshold 2.0 -fit-spectral-pol 3 -pol i -baseline-averaging "+str(wscleanBaseLineAv)+" -name nosources_atrous_uv"+str(l_name)+"_robust"+str(r_name)+"_taper"+str(wscleanT)+" -scale "+str(wscleanScale)+"arcsec -niter 10000 -multiscale -taper-gaussian "+str(wscleanT)
					#looping over various input directories and finding cleanCommand
					for inputDir in inputDirectory_arr:
						wscleancommand +=" "
						wscleancommand +=str(inputDir)
					#run cleanCommand
					print("Executing WSclean command without fakesource: "+str(wscleancommand))
					os.system(wscleancommand)
                    
					#move files to folder and folder to directory
					os.system("mv nosources_atrous_uv"+str(l_name)+"_robust"+str(r_name)+"_taper"+str(wscleanT)+"* nosources_atrous_UV"+str(l_name)+"_R"+str(r_name)+"_T"+str(wscleanT)) 
					os.system("mv nosources_atrous_UV"+str(l_name)+"_R"+str(r_name)+"_T"+str(wscleanT)+" UV"+str(l_name)+"_R"+str(r_name)+"_T"+str(wscleanT))
			else:
				#no taper-looping...
				#run actual wsclean with given parameters
				wscleancommand = "wsclean -no-update-model-required -multiscale -beam-size 20 -fits-mask newMask.mask.fits -minuv-l "+str(wscleanL)+".0 -size "+str(wscleanSize)+" "+str(wscleanSize)+" -reorder -weight briggs "+str(wscleanR)+" -weighting-rank-filter 3 -clean-border 1 -mgain 0.8 -fit-beam -data-column CORRECTED_DATA -join-channels -channels-out 6 -padding 1.4 -auto-mask 2.5 -auto-threshold 2.0 -fit-spectral-pol 3 -pol i -baseline-averaging "+str(wscleanBaseLineAv)+" -name nosources_atrous_uv"+str(l_name)+"_robust"+str(r_name)+" -scale "+str(wscleanScale)+"arcsec -niter 10000 -multiscale"
				#looping over various input directories and finding cleanCommand
				for inputDir in inputDirectory_arr:
					wscleancommand+=" "
					wscleancommand+=str(inputDir)
				#run cleanCommand
				print("Executing WSclean command without fakesource: "+str(wscleancommand))
				os.system(wscleancommand)
                
				#move files to folder and folder to directory
				os.system("mv nosources_atrous_uv"+str(l_name)+"_robust"+str(r_name)+"* nosources_atrous_UV"+str(l_name)+"_R"+str(r_name)) 
				os.system("mv nosources_atrous_UV"+str(l_name)+"_R"+str(r_name)+" UV"+str(l_name)+"_R"+str(r_name))
                
                
	####### W I T H   F A K E S O U R C E #######
	#preparation
	subtractPointSources(inputDirectory_arr, fakeSourceCatalog)
	#L-loop
	for wscleanL in uvCuts_arr:
		if len(str(wscleanL)) <3:
			l_name = "0"+str(wscleanL)
		else:
			l_name = str(wscleanL)
		#R-loop
		for wscleanR in robust_arr:
			if wscleanR < 0:
				r_name = "m"+str((-1)*wscleanR)
			else:
				r_name = str(wscleanR)
			if "." in r_name:
				r_name = r_name.replace('.', '')
			#for negative wightings, looping over tapers (entries in taper_arr)
			if wscleanR < 0:
				#T-loop
				for wscleanT in taper_arr:
					fakeSourceComb = "fakesource"+str(fakeSource[0])+"mJy_"+str(fakeSource[1])+"arcsec_"+str(fakeSource[2])+"_"+str(fakeSource[3])
					#run actual wsclean with given parameters for fake source
					wscleancommand = "wsclean -no-update-model-required -multiscale -beam-size 20 -fits-mask newMask.mask.fits -minuv-l "+str(wscleanL)+".0 -size "+str(wscleanSize)+" "+str(wscleanSize)+" -reorder -weight briggs "+str(wscleanR)+" -weighting-rank-filter 3 -clean-border 1 -mgain 0.8 -fit-beam -data-column CORRECTED_DATA -join-channels -channels-out 6 -padding 1.4 -auto-mask 2.5 -auto-threshold 2.0 -fit-spectral-pol 3 -pol i -baseline-averaging "+str(wscleanBaseLineAv)+" -name "+str(fakeSourceComb)+"_atrous_uv"+str(l_name)+"_robust"+str(r_name)+"_taper"+str(wscleanT)+" -scale "+str(wscleanScale)+"arcsec -niter 10000 -multiscale -taper-gaussian "+str(wscleanT)
					#looping over various input directories and finding cleanCommand
					for inputDir in inputDirectory_arr:
						wscleancommand+= " "
						wscleancommand+=str(inputDir)
					#run cleanCommand
					print("Executing WSclean command with fakesource: "+str(wscleancommand))
					os.system(wscleancommand)
					#move files to folder and folder to directory
					os.system("mv "+str(fakeSourceComb)+"_atrous_uv"+str(l_name)+"_robust"+str(r_name)+"_taper"+str(wscleanT)+"* "+str(fakeSourceComb)+"_atrous_UV"+str(l_name)+"_R"+str(r_name)+"_T"+str(wscleanT)) 

					os.system("mv "+str(fakeSourceComb)+"_atrous_UV"+str(l_name)+"_R"+str(r_name)+"_T"+str(wscleanT)+" UV"+str(l_name)+"_R"+str(r_name)+"_T"+str(wscleanT))    
			else:
				#no taper-looping...
				fakeSourceComb = "fakesource"+str(fakeSource[0])+"mJy_"+str(fakeSource[1])+"arcsec_"+str(fakeSource[2])+"_"+str(fakeSource[3])
				#run actual wsclean with given parameters for fake source
				wscleancommand = "wsclean -no-update-model-required -multiscale -beam-size 20 -fits-mask newMask.mask.fits -minuv-l "+str(wscleanL)+".0 -size "+str(wscleanSize)+" "+str(wscleanSize)+" -reorder -weight briggs "+str(wscleanR)+" -weighting-rank-filter 3 -clean-border 1 -mgain 0.8 -fit-beam -data-column CORRECTED_DATA -join-channels -channels-out 6 -padding 1.4 -auto-mask 2.5 -auto-threshold 2.0 -fit-spectral-pol 3 -pol i -baseline-averaging "+str(wscleanBaseLineAv)+" -name "+str(fakeSourceComb)+"_atrous_uv"+str(l_name)+"_robust"+str(r_name)+" -scale "+str(wscleanScale)+"arcsec -niter 10000 -multiscale"
				for inputDir in inputDirectory_arr:
					wscleancommand+= " "
					wscleancommand+=str(inputDir)
				#run wsclean command
				print("Executing WSclean command with fakesource: "+str(wscleancommand))
				os.system(wscleancommand)
                
				#move files to folder and folder to directory
				os.system("mv "+str(fakeSourceComb)+"_atrous_uv"+str(l_name)+"_robust"+str(r_name)+"* "+str(fakeSourceComb)+"_atrous_UV"+str(l_name)+"_R"+str(r_name)) 
				os.system("mv "+str(fakeSourceComb)+"_atrous_UV"+str(l_name)+"_R"+str(r_name)+" UV"+str(l_name)+"_R"+str(r_name))
	return combination_arr

#locates a ds9 region file in the folder of the script.
#make sure that the only ds9 region file in the folder is the one you want to use!
#@return region: string with the region information
#@return regionFileName: name of the region file
def findRegionFile():
	for file in os.listdir(os.getcwd()):
		if file.endswith(".reg") and not file.endswith(".ds9.reg"):
			regionsFileName = file
	r = open(regionsFileName, "r")
	for line in r.readlines():
		if 'circle' in line:
			region = line[line.find("(")+1 : line.find(")")].split(",")
	print(region)
	return region, regionsFileName

#decides whether we want to do manual gaussian folding 
#I used the WSClean routine for that, so make sure to remove the option in the
#wsclean command (not included as parameter) when activate this option! 
def doGaussianFolding(folding_do, inpDataPath, inpDataName, targetbmaj, targetbmin, targetTheta, targetDataName):
	if folding_do:
		targetDataName = gaussianFolding(inpDataPath, inpDataName, targetbmaj, targetbmin, targetTheta, targetDataName)
	return targetDataName
	
#manual folding routine that folds the input data
#I used the WSClean routine for that, so make sure to remove the option in the
#wsclean command (not included as parameter) when activate this option! 
#@param inpDataPath: path to the fits file that contains the data which has to be folded
#@param inpDataName: name of the fits file that contains the data which has to be folded
#@param targetbmaj: major beam axis on which the data has to be folded
#@param targetbmin: minor beam axis on which the data has to be folded
#@param targetTheta: target beam angle on which the data has to be folded
#@param targetDataName: name of the file containing the folded data
#@return folded data 
def gaussianFolding(inpDataPath, inpDataName, targetbmaj, targetbmin, targetTheta, targetDataName):
	targetSigmaX = targetbmaj/(2.*np.sqrt(2.*np.log(2.)))
	targetSigmaY = targetbmin/(2.*np.sqrt(2.*np.log(2.)))
    
	with fits.open(inpDataPath+'/'+inpDataName) as hdul:
		header = hdul[0].header
		data = hdul[0].data
		sigmaX = np.sqrt(np.abs(targetSigmaX**2 - (3600*header['BMAJ']/(2.*np.sqrt(2.*np.log(2.))))**2))
		sigmaY = np.sqrt(np.abs(targetSigmaY**2 - (3600*header['BMIN']/(2.*np.sqrt(2.*np.log(2.))))**2))
		theta = targetTheta - header['BPA']
		kernel = co.Gaussian2DKernel(x_stddev=sigmaX, y_stddev=sigmaY, theta = theta)
		data_fold = scipy_convolve(data[0][0][:][:], kernel)
		data_fold[:][:]*=((targetbmaj*targetbmin)/(header['BMAJ']*3600*header['BMIN']*3600))
		header['BMAJ']=targetbmaj/3600
		header['BMIN']=targetbmin/3600
		header['BPA'] = targetTheta
		if os.path.isfile(inpDataPath+'/'+targetDataName):
			os.remove(inpDataPath+'/'+targetDataName)
		hdu = fits.PrimaryHDU(header=header,data=data_fold)
		hdu.writeto(inpDataPath+'/'+targetDataName)
	return targetDataName


#main method for the ring integration. Creates all the plots
#@param stepsForRadialProfile: step size of the radial profile
#@param beamx, beamy: beamInformation
#@param FWHMFitMinimum: FWHM of the narrowest Gaussian for FWHM plot
#@param FWHMFitMaximum: FWHM of the wides Gaussian for FWHM plot
#@param FWHMFitSteps: step size between two fitting FWHMs for FWHM plot
#@param regionstring: string that contains region information
#@param comb: combination of UVcut, weighting and taper from the combination_array
#@param galFWHM: half light radius of the galaxy
#@param fakesource: array that contains fake source information
#@param folding_do: boolean whether do manual folding or not
#@param targetbmaj: major beam axis on which the data has to be folded
#@param targetbmin: minor beam axis on which the data has to be folded
#@param targetTheta: target beam angle on which the data has to be folded
def ringIntMain(stepsForRadialProfile, beamx, beamy, FWHMFitMinimum, FWHMFitMaximum, FWHMFitSteps, regionstring, comb, galFWHM, fakeSource, folding_do, targetbmaj, targetbmin, targetTheta):
	#converting the parameters into the right type..
	beamInformation = [beamx, beamy]
	stepForRadialProfile=float(stepsForRadialProfile)
	FWHMFitMinimum = float(FWHMFitMinimum)
	FWHMFitMaximum = float(FWHMFitMaximum)
	FWHMFitSteps = float(FWHMFitSteps)
	galFWHM = float(galFWHM)
	combname = comb.replace('_',' ')
	for file in os.listdir(os.getcwd()+'/'+comb):
		#find nosource data...
		if file.startswith("nosource"):
			for datasheet in os.listdir(os.getcwd()+'/'+comb+'/'+file):
				if datasheet.endswith("-MFS-image.fits"):
					fitsFileName = datasheet
					fitsFilePath = os.getcwd()+'/'+comb+'/'+file+'/'
					print(".fitsfile without fake source found:\n"+str(fitsFilePath)+str(datasheet))
					print("folding data from"+str(fitsFilePath)+str(fitsFileName))
					fitsFileName = doGaussianFolding(folding_do, fitsFilePath, fitsFileName, targetbmaj, targetbmin, targetTheta, comb+'_foldedData.fits')
					fitsFilePath = os.getcwd()+'/'+comb+'/'+file+'/'+fitsFileName
					print("Done!")
		#find fakesource data and store in list
		if file.startswith("fakesource"):
			for datasheet in os.listdir(os.getcwd()+'/'+comb+'/'+file):
				if datasheet.endswith("-MFS-image.fits"):
					fakefitsFileName = datasheet
					fakefitsFilePath = os.getcwd()+'/'+comb+'/'+file+'/'
					print(".fitsfiles with fake source found:\n"+str(fakefitsFilePath)+str(datasheet))
					print("folding data from"+str(fakefitsFilePath)+str(fakefitsFileName))
					fakefitsFileName = doGaussianFolding(folding_do, fakefitsFilePath, fakefitsFileName, targetbmaj, targetbmin, targetTheta, comb+'_foldedData.fits')
					fakefitsFilePath = os.getcwd()+'/'+comb+'/'+file+'/'+fakefitsFileName
					print("Done!")           
	print("region found:\n"+str(regionstring))
	#create output directory...
	if not os.path.exists(os.getcwd()+'/'+comb+'/output'):
		os.makedirs(os.getcwd()+'/'+comb+'/output')
	outputpath = os.getcwd()+'/'+comb+'/output/'
    

                                            
	#######CALCULATION ON OBSERVATIONAL DATA#######
	print("Data name: "+fitsFilePath)                                    
	h_real, d_real = loadFitsFile(fitsFilePath)

	wcs = loadWCS(h_real)
	regionInformation = loadRegionFile(regionstring,wcs=wcs,data_header=h_real)
	print("Loading information from regionsFile...\n"+"xcoordinate: "+str(regionInformation[0])+"\n"+"ycoordinate: "+str(regionInformation[1])+"\n"+"radius: "+str(regionInformation[2]))
	print("Calculating radial profile for observational data...")
	dist, flux, yerr = findRadialProfile(xc=regionInformation[0], yc=regionInformation[1], steps=stepsForRadialProfile, radius=regionInformation[2], data_arr=d_real, header=h_real, beamInformation=beamInformation)

	#beamfolding and rescaling (arcsec -> arcmin)
	dist_fr = rescaleArray(dist,(1./60.))
	flux_fr = rescaleArray(flux,10**6)
	yerr_fr = rescaleArray(yerr,10**6)
   
	#writing data to .txtfile 
	writeData(path=outputpath, name="observational_data", xdata=dist_fr, ydata=flux_fr, yerror=yerr_fr)
    
	#fit gaussian to observational data...
	print("Fitting Gaussians to observational data...\n")
	global binit
	binit=galFWHM/2.355
	popt, pcov = curve_fit(f=gaussian, xdata=dist, ydata=flux, sigma=yerr)
	fitted_a_real = popt[0]
	fitted_a_real_std = np.sqrt(pcov[0][0])
	fitted_a_real_fr = fitted_a_real*10**6
	fitted_a_real_std_fr = fitted_a_real_std*10**6
	print("Best fitting for a gaussian to obsercational data "+str(fitted_a_real_fr)+" +/- "+str(fitted_a_real_std_fr))
    
	#filling arrays for the plot
	bestfit_real = [] #flux values with gaussian distribution...
	spaceholder = []
	for i in dist:
		bestfit_real.append(gaussian(i,fitted_a_real))
		spaceholder.append("")
	#rescaling Array Jy -> microJy and folding beam..
	bestfit_real_fr = rescaleArray(bestfit_real,10**6)
    
	writeData(path=outputpath, name="gaussian_fit_for_observational_data", xdata=dist_fr, ydata=bestfit_real_fr, yerror=spaceholder)
    
    
	#######CALCULATIONS ON OBSERVATIONAL DATA + FAKE SOURCES#######
    
	h_fake, d_fake = loadFitsFile(fakefitsFilePath)
        
	#find radial profile for one fakesource...
	print("Calculating radial profile for observational data + fakesource "+str(fakeSource[0]+"mJy "+str(fakeSource[1])+"arcsec "+str(fakeSource[2])+" "+str(fakeSource[3])))
    
	dist_f, flux_f, yerr_f = findRadialProfile(xc=regionInformation[0], yc=regionInformation[1], steps=stepsForRadialProfile, radius=regionInformation[2], data_arr=d_fake, header=h_fake, beamInformation=beamInformation)
	
	#beamfolding and rescaling (arcsec -> arcmin)
	dist_f_fr = rescaleArray(dist_f,(1./60.))
	flux_f_fr = rescaleArray(flux_f,10**6)
	yerr_f_fr = rescaleArray(yerr_f,10**6)

        
	#write fitted data to .txtfile...
	writeData(path=outputpath, name="observational_data_+_fakesource_"+str(fakeSource[0])+"mJy_"+str(fakeSource[1])+"arcsec_"+str(fakeSource[2])+"_"+str(fakeSource[3]), xdata=dist_f_fr, ydata=flux_f_fr, yerror=yerr_f_fr)
    	
	#fit gaussian to observational + fakesource data
	popt, pcov = curve_fit(f=gaussian, xdata=dist_f, ydata=flux_f,sigma=yerr_f)
	fitted_a_fake = popt[0]
	fitted_a_fake_std = np.sqrt(pcov[0][0])
	fitted_a_fake_fr = fitted_a_fake*10**6
	fitted_a_fake_std_fr = fitted_a_fake_std*10**6
	print("Best fitting a for gaussian with this fakesource: "+str(fakeSource[0])+"mJy "+str(fakeSource[1])+"arcsec "+str(fakeSource[2])+" "+str(fakeSource[3])+": "+str(fitted_a_fake_fr)+" +/- "+str(fitted_a_fake_std_fr))
    
	#filling arrays for the plot	
	bestfit_fake = []
	spaceholder_f = []
	for i in dist_f:
		bestfit_fake.append(gaussian(i, fitted_a_fake))
		spaceholder_f.append("")
            
	bestfit_fake_fr = rescaleArray(bestfit_fake, 10**6)
	#write fitted data to .txtfile
	writeData(path=outputpath, name="gaussian_fit_for_observational_data_+_fake_source_"+str(fakeSource[0])+"_mJy_"+str(fakeSource[1])+"arcsec_"+str(fakeSource[2])+"_"+str(fakeSource[3]), xdata=dist_f_fr, ydata=bestfit_fake_fr, yerror=spaceholder_f)
    	


	#######PLOTTING RESULTS#######

	#Plotting radial profile...
	print("Plotting radial profile...")
	print("dist")
	print(dist)
	print("dist_fr:")
	print(dist_fr)
	fig, ax = plt.subplots()
	#observational plot with data points and fitted gaussian
	ax.plot(dist_fr, flux_fr, 'bs', label="Observation")
	ax.errorbar(x=dist_fr, y=flux_fr, yerr=yerr_fr, color='b',fmt='o', alpha=1, drawstyle='default', capsize=3)
	ax.plot(dist_fr, bestfit_real_fr, 'b--', label='Best fit gaussian with a = '+str(np.around(fitted_a_real_fr,1))+r' $\mu$Jy/beam $\pm$ '+str(np.around(fitted_a_real_std_fr,1))+r' $\mu$Jy/beam')
    
	#plot for observation + fakesources
	ax.plot(dist_f_fr, flux_f_fr, 'ro', label="Observation + fake source")
	ax.errorbar(x=dist_f_fr, y=flux_f_fr, yerr=yerr_f_fr, color='r', fmt='o', alpha=1, drawstyle='default', capsize=3)
	ax.plot(dist_fr, bestfit_fake_fr, 'r--', label='Best fit gaussian with a = '+str(np.around(fitted_a_fake_fr,1))+r' $\mu$Jy/beam $\pm$ '+str(np.around(fitted_a_fake_std_fr,1))+r' $\mu$Jy/beam')
          
	plt.xlabel(r'$\Theta$ [arcmin]')
	plt.ylabel(r'$I_\nu$ [$\mu$Jy/beam]')
	plt.legend(loc='upper right', fontsize='small',numpoints=1)
	ax.grid(True, which='both')
	plt.xlim(left=0)
	plt.title("Radial profile: "+combname)
	plt.savefig(outputpath+'radial_profile'+comb+'.png')
	plt.clf()
    
	#######FITTING FOR DIFFERENT FWHM#######
	#for observational data...
	FWHM_real, a_real, e_real = findFitProfile(FWHMmin=FWHMFitMinimum, FWHMmax=FWHMFitMaximum, steps=FWHMFitSteps, xdata=dist, ydata=flux, yerror = yerr)
	#folding...
	FWHM_real_fr = rescaleArray(FWHM_real, (1./60.))
	a_real_fr = rescaleArray(a_real, 10**6)
	e_real_fr = rescaleArray(e_real, 10**6)
    
	#plotting data...
	plt.plot(FWHM_real_fr, a_real_fr, 'b-', label="Observation")
	plt.fill_between(x=FWHM_real_fr, y1=np.subtract(a_real_fr,e_real_fr), y2=np.add(a_real_fr,e_real_fr), color='b', alpha = 0.2)
	writeData(path=outputpath, name="FWHM_plot_real", xdata=FWHM_real_fr, ydata=a_real_fr, yerror=e_real_fr)
          
	#for observational data + fakesource
	FWHM_fake, a_fake, e_fake = findFitProfile(FWHMmin=FWHMFitMinimum, FWHMmax=FWHMFitMaximum, steps=FWHMFitSteps, xdata=dist_f, ydata=flux_f, yerror = yerr_f)
	FWHM_fake_fr = rescaleArray(FWHM_real, (1./60.))
	a_fake_fr = rescaleArray(a_fake, 10**6)
	e_fake_fr = rescaleArray(e_fake, 10**6)
	writeData(path=outputpath, name="FWHM_plot_fake", xdata=FWHM_fake_fr, ydata=a_fake_fr, yerror=e_fake_fr)
        
	#Plotting data...
	plt.plot(FWHM_fake_fr, a_fake_fr, 'r-', label="Observation + fake source")
	plt.fill_between(x=FWHM_fake_fr, y1=np.subtract(a_fake_fr,e_fake_fr), y2=np.add(a_fake_fr,e_fake_fr), color='r', alpha = 0.2)

	plt.legend(loc='upper right', fontsize='small',numpoints=1)
	plt.xlabel("FWHM [arcmin]")
	plt.ylabel(r'a [$\mu$Jy/beam]')
	plt.grid(True, which='both')
	plt.title("FWHM-Plot: "+combname)
	plt.savefig(outputpath+'FWHM_plot'+combname+'.png')


#analyzes the beam of the data using the PSF output from WSClean
#@param combination: combination of UVcut, weighting and taper from the combination_array
#@param wscleanSize: Size of the Map
def findPSFDistribution(combination, wscleanSize):
	comb = combination.replace('_',' ')
	for file in os.listdir(os.getcwd()+'/'+str(combination)):
		if file.startswith('nosources'):
			for datasheet in os.listdir(os.getcwd()+'/'+combination+'/'+str(file)):
				if datasheet.endswith('-MFS-psf.fits'):
					name = datasheet
					path = os.getcwd()+'/'+combination+'/'+file+'/'+name
	print("Found PSF: "+ name)
	headerPSF, dataPSF = loadFitsFile(path)
	wcs = loadWCS(headerPSF)
	#implement 0.5*wscleanSize as centerpoint and some radius?
	regionInformationPSF=[0.5*wscleanSize,0.5*wscleanSize,200]
	beamInformationPSF=[headerPSF['BMAJ'], headerPSF['BMIN']]
	distPSF, fluxPSF, yerrPSF = findRadialProfilePSF(xc=regionInformationPSF[0],yc=regionInformationPSF[1],steps=1, radius=regionInformationPSF[2], data_arr=dataPSF, header=headerPSF, beamInformation=beamInformationPSF)
    

	fig, ax = plt.subplots()
	ax.plot(rescaleArray(distPSF,(1./60)), fluxPSF, 'b-', label='Observation')
	ax.errorbar(x=rescaleArray(distPSF,(1./60)), y=fluxPSF, yerr=yerrPSF, color='b', alpha=0.5, drawstyle='default', capsize=3)
	plt.xlabel('radius [arcmin]')
	plt.ylabel('PSF [Jy/beam]')
	plt.title('Weighting for '+comb)
	plt.legend(loc='upper right')
	plt.xlim(left=-0.1, right=2.6)
	plt.grid(True)
	plt.savefig(combination+'/output/PSF'+combination+'.png')
	plt.clf()
          
#converts string to an list of strings
#@param inputString: string that will be converted
#@return: list that contains strings
def interpretAsArray(inputString):
	out = inputString.replace(']','').replace('[','').replace(' ', '').replace('"','').split(',')
	return out
    
#converts string to a list of integers
#@param inputString: string that will be converted
#@return: list of integers
def interpretAsIntList(inputString):
	out = []
	inputString = inputString.replace(']','').replace('[','').replace(' ', '').replace('"','').split(',')
	i = 0
	while i < len(inputString):
		out.append(int(inputString[i]))
		i+=1
	return out

    
#converts string to a list of floats
#@param inputString: string that will be converted
#@return: list of floats
def interpretAsFloatList(inputString):
	out = []
	inputString = inputString.replace(']','').replace('[','').replace(' ', '').replace('"','').split(',')
	i = 0
	while i < len(inputString):
		out.append(float(inputString[i]))
		i+=1
	return out

#main method.
def main():
	print("load parameter sheet..")
	initFitsFileName, fakeSource_arr, uvCuts_arr, robust_arr, taper_arr, wscleanSize, wscleanBaseLineAv, beamInformation, FWHMFitMinimum, FWHMFitMaximum, FWHMFitSteps, inputDirectory_arr, wscleanScale, galFWHM, folding_do, targetbmaj, targetbmin, targetTheta = loadOptions()
	print("Done!")
	#rewrite input strings as arrays when needed..
	fakeSource_arr = interpretAsArray(fakeSource_arr)
	uvCuts_arr = interpretAsIntList(uvCuts_arr)
	robust_arr = interpretAsFloatList(robust_arr)
	taper_arr = interpretAsIntList(taper_arr)
	wscleanSize = int(wscleanSize)
	wscleanBaseLineAv = float(wscleanBaseLineAv)
	beamInformation = interpretAsFloatList(beamInformation)
	FWHMFitMinimum = float(FWHMFitMinimum)
	FWHMFitMaximum = float(FWHMFitMaximum)
	FWHMFitSteps = float(FWHMFitSteps)
	inputDirectory_arr = interpretAsArray(inputDirectory_arr)
	wscleanScale = float(wscleanScale)
	galFWHM = float(galFWHM)
	folding_do = bool(folding_do)
	targetbmaj = float(targetbmaj)
	targetbmin = float(targetbmin)
	targetTheta = float(targetTheta)
	print("Done!")
	print("identifying point sources with PyBDSF..")
	#pointSourceCatalog = findPointSources(fitsFileName=initFitsFileName)
	pointSourceCatalog = 'scale02_rms_66_15.bbs'
	print("Done!")
	print("storing fake sources...")# given in fakeSource_arr. For list structure see function comment
	fakeSourceCatalog = storeFakeSource(pointSourceCatalog=pointSourceCatalog, fakeSource=fakeSource_arr)
	print("Done!")
	print("run wsClean..")
	#combination_array = runWSClean(uvCuts_arr=uvCuts_arr, robust_arr=robust_arr, taper_arr=taper_arr, fakeSource=fakeSource_arr, wscleanSize=wscleanSize, wscleanBaseLineAv=wscleanBaseLineAv, inputDirectory_arr=inputDirectory_arr, pointSourceCatalog=pointSourceCatalog, fakeSourceCatalog=fakeSourceCatalog, wscleanScale=wscleanScale)
	
	combination_array = runWSClean(uvCuts_arr=uvCuts_arr, robust_arr=robust_arr, taper_arr=taper_arr, fakeSource=fakeSource_arr, wscleanSize=wscleanSize, wscleanBaseLineAv=wscleanBaseLineAv, inputDirectory_arr=inputDirectory_arr, pointSourceCatalog=pointSourceCatalog, fakeSourceCatalog=fakeSourceCatalog, wscleanScale=wscleanScale)
    
	#no wsclean/PyBDSF
	#combination_array = []
	#for file in os.listdir(os.getcwd()):
	#	if file.startswith('UV'):
	#		combination_array.append(file)
	#print("Found the following combinations...\n")
	#for comb in combination_array:
	#	print(str(comb))

	print("Done!")
	print("Loading regions file...")
	region, regionsFileName = findRegionFile()
	print("RegionsFile "+str(regionsFileName)+" found! ("+str(region)+")")
	print("Run Main Ring integration..")
	for combination in combination_array:
    		
		print("Calculating in: "+str(combination))
		ringIntMain(FWHMFitSteps, beamInformation[0], beamInformation[1], FWHMFitMinimum, FWHMFitMaximum, FWHMFitSteps, region, combination, galFWHM, fakeSource_arr, folding_do, targetbmaj, targetbmin, targetTheta)
		print("Calculating PSF in: "+str(combination))
		findPSFDistribution(combination, wscleanSize)

    
	#ringIntMain(FWHMFitSteps, beamInformation[0], beamInformation[1], FWHMFitMinimum, FWHMFitMaximum, FWHMFitSteps, region, 'UV160_R02', galFWHM, fakeSource_arr, folding_do, targetbmaj, targetbmin, targetTheta)
	#findPSFDistribution('UV160_R02', wscleanSize)

if __name__ == '__main__':
	main()

