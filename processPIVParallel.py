# GBell edit: adding dpiv
# Writen by Graham Bell 16/02/2016
# Program to take a list of separate image pairs and prepare processing for PIVview

import subprocess
import glob
import re
import multiprocessing
from functools import partial
import compileNC
import h5py
import os, os.path
import PIVImagepreprocessing
from skimage import io
import matplotlib.pyplot as plt
import tarfile
import datetime
import PIVCasesInfo
import pickle
import numpy as np

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
def GetPIVpairs(currentFolder, bkgndImgKeyword = None):

    # Check that a folder exists.
    if os.path.isdir(currentFolder) == False:
        print("Error: Folder: ", currentFolder, " does not exist")
        exit(1)

    filenames = glob.glob(currentFolder + '*.tif')
    # Sort the filenames by frame number.
    filenames.sort(key=natural_keys)
    # Check that there is an even number of frames
    if len(filenames) % 2 != 0:
        print("Odd number of file names detected for folder:")
        print(len(filenames))
        print(folder)
        exit(1)

    if bkgndImgKeyword is not None:
        # Check for a median image even (0) and odd (1)
        bkgndImages = [s for s in filenames if bkgndImgKeyword in s]
    else: bkgndImages = []

    if len(bkgndImages) > 0:
        # Remove them from the filenames list.
        for name in bkgndImages:
            filenames.remove(name)

    # Create a list of the image pairs.
    PIVpairs = []
    numOfPairs = len(filenames)/2

    for i in range(0, len(filenames), 2):
        pairtuple = (filenames[i], filenames[i + 1])
        PIVpairs.append(pairtuple)
    if len(PIVpairs) < 1:
        print("Error: No PIV image pairs found for folder:")
        print(currentFolder)
    return PIVpairs, bkgndImages

def mp_PIVworker(pair): #, currentFolder):
    printpair0 = pair[0]
    printpair1 = pair[1]
    print('Solving pair: ', printpair0, ' ', printpair1, datetime.datetime.now())
    command = " ".join(['dpiv', paramFile, pair[0], pair[1]])
    subprocess.call(['/bin/bash', '-i', '-c', command])
    #subprocess.call(['/bin/bash', '-c', command])

def mp_SubtractImages(pair, evenBkgndImage, oddBkgndImage, subtractImgBaseName, subtractFolder, overWriteSubFiles):

    imageName0 = pair[0]
    imageName1 = pair[1]

    imageNum0 = imageName0[-12:]
    imageNum1 = imageName1[-12:]
    # Remove the word 'Sequence' from the filename and replace with "SUBD"
    SUBDimageName0 = subtractFolder + subtractImgBaseName + imageNum0
    SUBDimageName1 = subtractFolder + subtractImgBaseName + imageNum1

    # Check that the files don't exist already. Don't do double the work.
    if not os.path.isfile(SUBDimageName0) or overWriteSubFiles is True:
        img0 = io.imread(imageName0, as_grey=True)
        img0 = PIVImagepreprocessing.subtractBackground(img0,evenBkgndImage)
        # Image comes back as a 16 bit array scaled from 0 to 2^16.
        # Ready for saving as 16bit tiff.
        io.imsave(SUBDimageName0, img0)
        print("Subtracted:", (subtractImgBaseName + imageNum0), datetime.datetime.now())
    else: print("File exists", SUBDimageName0,"Skipping:", datetime.datetime.now())
    if not os.path.isfile(SUBDimageName1) or overWriteSubFiles is True:
        img1 = io.imread(imageName1, as_grey=True)
        img1 = PIVImagepreprocessing.subtractBackground(img1,oddBkgndImage)
        io.imsave(SUBDimageName1, img1)
        print("Subtracted:", (subtractImgBaseName + imageNum1), datetime.datetime.now())
    else: print("File exists", SUBDimageName1, "Skipping:", datetime.datetime.now())

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def mp_compressTar(packedTarAndPIVPairs):
    tarName = packedTarAndPIVPairs[0]
    PIVpairsRawImages = packedTarAndPIVPairs[1]
    print("Compressing raw images to:", tarName, datetime.datetime.now())
    tar = tarfile.open(tarName, "w:bz2")
    for (image0, image1) in PIVpairsRawImages:
        tar.add(image0)
        print("Compressed:", image0, datetime.datetime.now())
        tar.add(image1)
        print("Compressed:", image1, datetime.datetime.now())
    tar.close()
    print("Done file:",tarName)

def histPlotter(data, numBins, title):
    hist, bins = np.histogram(data, numBins)
    print(hist, bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, np.log10(hist), align='center', width=width)
    plt.xlabel('value')
    plt.ylabel('log10(counts)')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    picklePlace = 'AlkislarStats/'
    # Case and set information is saved in a pickled dictionary
    [case1, case2, case3, case4, case5, case6, case7, case8, case9] = PIVCasesInfo.cases
    #-----------------Input Options ----------------------------
    cases = [case1, case2, case3, case4, case5, case6, case7, case8, case9]

    #cases = [case1, case4]
    cases = [case4]
    for case in cases:

        subtractBkgnd =         False
        overWriteSubFiles =     False
        workOnSubImages =       True
        doPIV =                 False
        compileH5 =             True
        compressRawImages =     False
        #----------------Input Options -----------------------------

        imagesFolder = [PIVCasesInfo.setPath(setNumber) for setNumber in case['sets']]
        GrahamDrive0 = '/media/graham/GrahamDrive0/Far2016/'
        paramFile = GrahamDrive0 + 'ServerParams24.par'

        for index, folder in enumerate(imagesFolder):
            subtractFolder = 'SUBD/'
            currentFolder = folder
            rawFolder = currentFolder
            subtractFolder = rawFolder + subtractFolder
            print("Folder: ", currentFolder, ":", datetime.datetime.now())
            os.makedirs(subtractFolder, exist_ok=True)

            if subtractBkgnd is True:
                PIVpairs, bkgndImages = GetPIVpairs(rawFolder, bkgndImgKeyword='medianImage')

                if len(PIVpairs) > 1 and len(bkgndImages) > 0:
                    # Get the first occurance of
                    evenBkgndImage = [s for i, s in enumerate(bkgndImages) if '0.tif' in s][0]
                    oddBkgndImage = [s for i, s in enumerate(bkgndImages) if '1.tif' in s][0]
                    evenBkgndImage = io.imread(evenBkgndImage,as_grey=True)
                    oddBkgndImage = io.imread(oddBkgndImage, as_grey=True)

                    #subtract each image from the background and save.
                    subtractImgBaseName = 'subd'
                    func = partial(mp_SubtractImages, evenBkgndImage=evenBkgndImage, oddBkgndImage=oddBkgndImage,
                                   subtractImgBaseName=subtractImgBaseName, subtractFolder=subtractFolder,
                                   overWriteSubFiles=overWriteSubFiles)
                    # Set up for parallel processing; use 7 processors.
                    p = multiprocessing.Pool(7)
                    p.map_async(func, PIVpairs)
                    p.close()
                    p.join() # Wait unit all processes are done before continuing

            if doPIV is True:
                if workOnSubImages is True:
                    PIVpairs, _ = GetPIVpairs(subtractFolder)
                else:
                    PIVpairs, _ = GetPIVpairs(rawFolder)
                if len(PIVpairs) > 1:
                    p = multiprocessing.Pool(7)
                    p.map(mp_PIVworker, PIVpairs)
                    p.close()
                    p.join()

            if compileH5 is True:
                if workOnSubImages is True:
                    PIVpairs, _ = GetPIVpairs(subtractFolder)
                    compiledFileName = subtractFolder + 'compiledNC.h5'
                else:
                    PIVpairs, _ = GetPIVpairs(rawFolder)
                    compiledFileName = rawFolder + 'compiledNC.h5'
                if len(PIVpairs) > 1:
                    Uarray, Varray, flagArray = compileNC.makeNC(subtractFolder)
                    print('Done reading files.')
                    print('Opening h5 file:', compiledFileName, datetime.datetime.now())
                    h5f = h5py.File(compiledFileName, 'w')
                    print('Writing datasets...')
                    # Let h5py auto chunk as I'll be using both temporal fields and spatial fields.
                    # This results in no preferred chunking direction.
                    h5f.create_dataset('axialVel', data=Uarray.astype(np.float32), chunks=True, compression=None)
                    h5f.create_dataset('transverseVel', data=Varray.astype(np.float32), chunks=True, compression=None)
                    h5f.create_dataset('flags', data=flagArray.astype(np.int8), chunks=True, compression=None)
                    print('Closing file...')
                    h5f.close()
                    print('Done.', datetime.datetime.now())
                    deleteables = ['Uarray', 'Varray', 'flagArray']
                    for deleteable in deleteables:
                        if deleteable in locals():
                            del deleteable
            if compressRawImages is True:
                PIVpairsRawImages, _ = GetPIVpairs(rawFolder)
                if len(PIVpairsRawImages) > 1:
                    # Break up the tar chunk into 5 tar files. And save each as Xof5.tar.bz2
                    setName = folder[:-2]
                    print("Compressing set:", setName, datetime.datetime.now())
                    numOfTars = 5
                    parts = range(numOfTars)
                    tarFileNames = [currentFolder + 'Set' + str(case['sets'][index]) + "RawImages" + str(i+1) + "of" + str(numOfTars) + ".tar.bz2" for i in parts]
                    PIVpairsChunked = list(chunks(PIVpairsRawImages, int(len(PIVpairsRawImages)/numOfTars)))
                    # Pack the tarName and PIV images together as tuple and unpack inside each process
                    packedTarPIVpairs = [(tarName, pivSet) for tarName, pivSet in zip(tarFileNames, PIVpairsChunked)]

                    # For some reason only map works here.
                    p = multiprocessing.Pool(5)
                    p.map(mp_compressTar, packedTarPIVpairs)
                    p.close()
                    p.join()
                    print(PIVpairsChunked)
