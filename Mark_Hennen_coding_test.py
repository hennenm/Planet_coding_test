#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:00:11 2022

@author: markhennen
"""
# ============================================================================================
# --------------------------- Coding Challenge: Mark Hennen ----------------------------------
# This Python function takes a multi-band image and single band annotated layer of the same 
# dimensions and geolocation, returning a python dictionary of training (80%) and testing (20%)
# grid cells of specified size (64 * 64 pixels to match ML patch). This function addresses each  
# of the requirements of the challenge, described throughout the code. 
# Cell size can be changed on line <65>. 
# --------------------------------------------------------------------------------------------
# Results are produced as arrays, but can be converted back into .tif images with original 
# spatial reference system using the function ArrayToImage. This is optional. 
# ===========================================================================================
# For demonstration, there is a small script after the functions. Here, you can define your 
# working directory and run the TrainSample function and upload the results.
# You can optionally converted the resulting training and testing arrays into .tif images  
# using the ArrayToImage function on lines <239-241>.
# ============================================================================================

# Import required Python libraries
import os
import numpy as np
import pandas as pd
from osgeo import gdal, osr
from random import sample



def TrainSample (image, labels):
    # Read the image and labels, the filenames of which are provided to the function as input;
    # =========================================================================================
    img = gdal.Open(image)      # Read image data 
    lab = gdal.Open(labels)     # Read label data
    
    # Identify bandwise mean and standard deviation for each image band, as global constants  
    globalDat = {'mean': [],'std': []}     #empty global constant dictionary

    # Loop through each band, retrieve global constants in global dictionary 
    for band in range(1,img.RasterCount+1):
        data = img.GetRasterBand(band).ReadAsArray().astype('float')
        globalDat['mean'].append(np.mean(data))
        globalDat['std'].append(np.std(data))
    
    # Create an appropriate sampling grid of the image data;
    # ======================================================
    # Retrieve geotransformation info of input image
    gt = img.GetGeoTransform()
    # Retrieve spatail reference system
    prj=img.GetProjection()
    srs=osr.SpatialReference(wkt=prj)
    srs.AutoIdentifyEPSG()
    epsg = (str(srs.GetAuthorityCode(None)))
    # get coordinates of upper left corner
    xmin = gt[0]
    ymax = gt[3]
    # determine total length of raster
    xlen = img.RasterXSize
    ylen = img.RasterYSize
    # size of a single tile
    gsize = 64
    # number of tiles in x and y direction (when using exhaustive sampling of unique grid cells)
    xdiv = int(round(xlen/gsize, 0))
    ydiv = int(round(ylen/gsize, 0)) 
    # create lists of x and y coordinates
    xsteps = [xmin + gsize * i for i in range(xdiv+1)]
    ysteps = [ymax - gsize * i for i in range(ydiv+1)]
    # ----- Loop through each of the cells for further analysis ----------
    # Empty list for storing cell arrays
    gridDict = {}
    # loop over min and max x and y coordinates, for image segmentation
    for i in range(xdiv):
        for j in range(ydiv):
            # Define grid co-ordinates
            xmin = xsteps[i]    
            xmax = xsteps[i+1]  
            ymax = ysteps[j]    
            ymin = ysteps[j+1]    
            # Segment annotated band using gdal.warp, save as temporary file 
            gdal.Warp("labTile.tif", lab,
                      outputBounds = (xmin, ymin, xmax, ymax))
            # Read cell into memory
            labTile = gdal.Open("labTile.tif")
            # Get geotransform information of tile 
            gt = labTile.GetGeoTransform()
            
            # Ignore grid cells that do not contain one of the two target classes;
            # =================================================================            
            labTile = gdal.Open("labTile.tif").ReadAsArray() # Convert cell.tif image into an array
            # Define requred lulc class
            labGroup = [0,1]   
            # Test if recquired lulc labels occur in cell
            result = all(elem in labTile for elem in labGroup)
            
            if result: # All required classes occur in cell
                
                # Create multiclass label as the count of each label in a grid cell;            
                # =================================================================
                # Count number of each label in cell
                unique, counts = np.unique(labTile, return_counts=True)
                # Multiclass label in format Grid<#x+ycoordinat>//<#label>:count/<#label>:count/......
                gridLab = 'Grid:'+str(i)+str(j)+'//'
                #Loop through each class, adding count to multiclass label
                for CLASS in range(len(unique)):    
                    gridLab = gridLab+str(unique[CLASS])+':'+str(counts[CLASS])+'/'
                
                # Ppply bandwise zero-centering and scaling against some constants BANDWISE_MEAN and
                # BANDWISE_STD , respectively (these can be assumed to be global variables that 
                # are arrays of the same length as the number of spectral bands, i.e. four);           
                # ==================================================================================
                # segment image using grid coordinates             
                gdal.Warp("imgTile.tif", img, 
                          outputBounds = (xmin, ymin, xmax, ymax), dstNodata = -9999)
                # Read multispctral image
                imgTile = gdal.Open("imgTile.tif")
                # Loop through all bands and perform zero centring and scaling
                bands = []
                for band in range(1, imgTile.RasterCount+1): 
                    data = imgTile.GetRasterBand(band).ReadAsArray().astype('float')
                    data = (data-globalDat['mean'][band-1])/globalDat['std'][band-1]
                    bands.append(data)
                # Stack band arrays into single 4D array: 
                # Band 1 = stack[0], Band 2 = stack[1], Band 3 = stack[2], Band 4 = stack[3]
                stack = np.stack(bands, axis=0)
                
                # Store 4 band array, label band and geotransform info in a dictionary 
                # Dictionary structure: 
                # Multiclass label: 4 band array (position [0]), label band (position [1]), geotransform info (position [2])
                gridDict[gridLab] = stack, labTile, gt, epsg 
                
            else: # NOT all required classes occur in cell; ignore cell
                continue        
     
    # Split the cells and their labels into training (~80%) and test (~20%) sets;       
    # ===========================================================================
    # Determine the number of cells = 80%
    trainLen   = int(round(round(len(gridDict.keys()),0)*0.8,0))
    testLen    = len(gridDict)-trainLen
    
    # Sample 80% dictionary keys
    trainKeys  = sample(gridDict.keys(), trainLen) 
    # Store training dictionary from selction of trainig keys
    trainData  = {k:v for (k,v) in gridDict.items() if k in trainKeys}
    # Store remaining cells in testing (validation) dictionary 
    testData   = {k:v for (k,v) in gridDict.items() if k not in trainKeys}
    
    # Return the training and test sets and their respective labels
    # =============================================================
    # Save training and testing data as arrays within a dictionary (.npy) along with resepctive lables.
    np.save('Training_data', trainData)
    np.save('Validation_data', testData)
    # Describe results and location of datasets
    return print(f'Process complete \nNumber of Training cells:{trainLen} \nNumber of Testing cells: {testLen} \nTraining data saved as: Training_data.npy \nTesting data saved as: Validation_data.npy') 



# ============================================================================================================
# ----------------------- Optional function to convert array back to image -----------------------------------
# Convert arrays in results dictionary back to image files
def ArrayToImage (Arrdata, group): 
    # Two arguments:
    # Arrdata = dictionat containing arrays, 
    # group = string to preface output files, indicate training ('t') or validation ('v')
    grids = [x for x in Arrdata.keys()] #Produce list of grid cells to loop over
    # First loop, converts label band into .tif image
    for GRID in grids:
        imgid = group+GRID.split('//')[0]   # String for label layer output
        gt = Arrdata[GRID][2]               # Geotransform information
        res = gt[1]                         # Pixel resolution
        srs = Arrdata[GRID][3]              # Spatial reference information
        # Get upper ledt co-ordinate of array
        xmin = gt[0]                        # x min co-ordinate
        ymax = gt[3]                        # y max co-ordinate
        # Extract label array data from results dictionary
        ar = Arrdata[GRID][1]
        # Determein size of cell (in pixels)
        xsize = len(ar[0])                  
        ysize = len(ar[1])
        # Identify central co-ordinates of upper left pixel
        xstart = xmin +res/2
        ystart = ymax - res/2
        # Create columns of X and Y coordinates for entire array 
        x = np.arange(xstart, xstart+xsize*res, res)
        y = np.arange(ystart, ystart-ysize*res, -res)
        x = np.tile(x, ysize)
        y = np.repeat(y, xsize)
        # Flatten array into 1-D series (row by row)
        flat = ar.flatten()                 
        # Combine flattened array with X,Y co-ordinates into dataframe
        lab = pd.DataFrame({"x":x, "y":y, "value":flat})
        lab = lab.reset_index(drop=True) # Remove index from df
        lab.to_csv(f'lab.xyz', index = False, header = None, sep = " ") # Store df as csv
        # Export labels dataframe as .tif file using gdal.Translate in original SRS.
        labtif = gdal.Translate(f'{imgid}_label.tif', f'lab.xyz', outputSRS = f'EPSG:{srs}')
        # Close open image
        labtif = None
        # Second loop, converts each band into an image then stacks into mutispectral image
        for BAND in range(len(Arrdata[GRID][0])):
            # Extract band array data from results dictionary and flatten into series
            ar = Arrdata[GRID][0][BAND]
            flat = ar.flatten()
            # Combine flattened array with X,Y co-ordinates into dataframe
            bandn = pd.DataFrame({"x":x, "y":y, "value":flat})
            bandn = bandn.reset_index(drop=True) # Remove index from df
            bandn.to_csv(f'bandn{BAND}.xyz', index = False, header = None, sep = " ") # Store df as csv
            # Export band dataframe as .tif file using gdal.Translate in original SRS.
            bandtif = gdal.Translate(f'bandn_{BAND}.tif', f'bandn{BAND}.xyz', outputSRS = f'EPSG:{srs}')
            
        # Temporary stacking directory
        outvrt = '/vsimem/stacked.vrt'
        # load each band image (.tif) file into a list
        bandgrp = ['bandn_0.tif', 'bandn_1.tif', 'bandn_2.tif', 'bandn_3.tif']
        # Stack bands as a virtual raster 
        outds = gdal.BuildVRT(outvrt, bandgrp, separate=True)
        # Export multispectral image as .tif file using gdal.Translate.
        outds = gdal.Translate(f'{imgid}_img.tif', outds)
        # Close open multispectral image
        outds = None


# ============================================================================================================
# ----------------------- Run these lines to use train_sample function ---------------------------------------
# Specify working directory
os.chdir ('/Volumes/Extreme SSD/Planet/')

# Run training sample function (input mutispectral.tif, label.tif)
TrainSample('insert_mutlispec_file.tif', 'insert_label_file.tif')

# Load training and test grid cell arrays, with associated lulc label array 
traindata = np.load('Training_data.npy', allow_pickle = 'TRUE').item()
testdata = np.load('Validation_data.npy', allow_pickle = 'TRUE').item()

#### ----------- Optional: Convert arrays into tif images ---------------------------------------------------- 
# Produce training images to directory
ArrayToImage(traindata,'t')
# Produce test images to directory
ArrayToImage(testdata,'v')









