# Routines to subtract the background and improve the image quality of the tiff files.
#Written by Graham Bell 15/04/16
import numpy as np
from skimage import io, exposure
import random
import glob
import os
import matplotlib.pyplot as plt

# Make the minimum of the image 0 and the max 1 and scale in between.
def normaliseImage(arr):
    arr = (-np.min(arr) + arr)
    arr = arr/np.max(arr)
    return arr


def subtractBackground(img, bkgnd):
    # Subtract the background from the image. Pass in a skikit image

    # Do adaptive histogram equalization to normalise the image.
    img = exposure.equalize_adapthist(img, clip_limit=0.03)

    bkgnd = exposure.equalize_adapthist(bkgnd, clip_limit=0.03)
    subd = img - bkgnd

    # Normalise the image from 0 to 1 then run as type to make 0 to 2^16 bit.
    subd = normaliseImage(subd)
    subd = (2**16)*subd
    subd = subd.astype('uint16')
    return subd


# Pass in the files to calculate the median image and return. Best use of memory
def getMedianImage(files):
    sampleImage = io.imread(files[0])
    imagesArray = np.zeros((sampleImage.shape[0],sampleImage.shape[1],len(files)), dtype='uint16')
    del sampleImage
    for index, value in enumerate(files):
        imagesArray[:,:,index] = io.imread(value)
    return np.median(imagesArray,axis=2).astype('uint16')


if __name__ == '__main__':

    #imagesFolder = ['Set77/','Set76/','Set59/','SampleBackgrounds/']
    # Get the filenames
    GrahamDrive0 = '/media/graham/GrahamDrive0/TwinTestImages/Far2016/'
    GrahamDrive1 = '/media/graham/GrahamDrive1/Far2016/'
    print(os.listdir(GrahamDrive0))
    for path in [GrahamDrive0,GrahamDrive1]:
        for object in os.listdir(path):
            if object.startswith('Set'):

                evens = ['*0.tif','*2.tif','*4.tif','*6.tif','*8.tif']
                odds = ['*1.tif','*3.tif','*5.tif','*7.tif','*9.tif']
                evenFiles = []
                oddFiles = []
                for even, odd in zip(evens, odds):
                    evenFiles += glob.glob(path + object + '/' + even)
                    oddFiles += glob.glob(path + object + '/' + odd)

                doMedian = True
                if doMedian == True:
                    # Create the medians of both images.
                    random.shuffle(evenFiles)
                    random.shuffle(oddFiles)
                    numofImages = 100
                    evenMedian = getMedianImage(evenFiles[:numofImages])
                    oddMedian = getMedianImage(oddFiles[:numofImages])
                    # Save out the images
                    io.imsave(path + object + '/' + 'medianImage0.tif',evenMedian.astype('uint16'))
                    io.imsave(path + object + '/' + 'medianImage1.tif',oddMedian.astype('uint16'))
                    print("Done and saved median:",(path + object + '/' + 'medianImage0.tif'))




    # Grab only images that end in a zero. Therefore even.

    #bkgndMedian = np.median(imagesArray,axis=2).astype('uint16')
    #images = glob.glob(GrahamDrive1 + imagesFolder[2] + evenOrOdd)
    #pivImage = io.imread(images[0])
