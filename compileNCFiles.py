from netCDF4 import Dataset
import numpy as np
import glob
import re

# ---- For sorting nicely.
# Code from: http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
# ------------------------

def makeNC(currentFolder):
    filenames = glob.glob(currentFolder + '*.nc')
    # Sort the file names
    # Sort the filenames by frame number.
    filenames.sort(key=natural_keys)
    # Get a sample file for shape
    ncFile = filenames[0]
    fh = Dataset(ncFile, mode='r')

    sample = fh.variables['piv_data'][0]
    sampleshape = sample.shape
    # Return 2 arrays; 1 for u and 1 for v
    Uarray = np.zeros((sampleshape[0],sampleshape[1],len(filenames)))
    Varray = np.zeros((sampleshape[0],sampleshape[1],len(filenames)))
    flagArray = np.zeros((sampleshape[0],sampleshape[1],len(filenames)),dtype=np.int8)
    del sample
    for index,file in enumerate(filenames):
        ncFile = file
        fh = Dataset(ncFile, mode='r')
        pivdata = fh.variables['piv_data'][0]
        flagdata = fh.variables['piv_flags'][0]
        Uarray[:, :, index] = pivdata[:, :, 0]
        Varray[:, :, index] = pivdata[:, :, 1]
        flagArray[:, :, index] = flagdata
        fh.close()
        print("Read: ", ncFile)
    return Uarray, Varray, flagArray

def getNCCoords(ncFile):
    fh = Dataset(ncFile, mode='r')

    imageCoordMax = fh.variables['piv_data'].coord_max
    imageCoordMin = fh.variables['piv_data'].coord_min

    pivStepx = fh.ev_IS_grid_distance_x
    pivStepy = fh.ev_IS_grid_distance_y

    pivPixelCoordsCols = np.arange(imageCoordMin[0], imageCoordMax[0] + pivStepy, pivStepy)
    pivPixelCoordsRows= np.arange(imageCoordMin[1], imageCoordMax[1] + pivStepx, pivStepx)
    return pivPixelCoordsRows, pivPixelCoordsCols

if __name__ == "__main__":
    GrahamDrive0 = '/media/graham/GrahamDrive0/TwinTestImages/Far2016/'
    imagesFolder = ['Set10/']

    # Get a list of all the files
    currentFolder = GrahamDrive0 + imagesFolder[0]
    U, V = makeNC(currentFolder)



