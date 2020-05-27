import h5py
import numpy as np
import dask.array as da
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import compileNC
import PIVCasesInfo
from PIVCasesInfo import cases
import ideallyExpandedTools
import datetime
import sys
import scipy.signal
import itertools
import os
sys.path.append("../PlotTools/")
# Custom plotting tools
import latexPlotTools as lpt
import collections
import itertools as it

import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools

def weightedAvgInterp(adjactentValues, diagonalValues):
    # Computes the weighted average for a nan value.
    # Takes the weighted average of the vector values surrounding it.
    diagonalWeight = np.sqrt(2) - 1.0
    adjacentWeight = 1.0
    weightedAvg = (np.sum(adjactentValues) * adjacentWeight + np.sum(diagonalValues)*diagonalWeight)/(len(adjactentValues)+len(diagonalValues))
    return weightedAvg

def validateStd(arr, temporalStd, temporalMean, stdCutOff):
    # Use numpy broadcasting to validate all in one go for max speed.
    # goodFields are marked with a 1.
    # Goodfields will be 1 if they are greater than the lower standard deviation
    print('Calculating greater')
    # The mean is a 2D : convert to 1 axis 3D array.
    temporalMean = temporalMean[:, :, np.newaxis]
    temporalStd = temporalStd[:, :, np.newaxis]

    #plot1DHist(x=vectorArray[146, 417, :], numBins=100)
    goodFieldNeg = np.greater(arr, temporalMean - stdCutOff * temporalStd)
    # Good fields will be 1 if they are less than the upper standard deviation
    print('Calculating lesser')

    goodFieldPos = np.less(arr, temporalMean + stdCutOff * temporalStd)
    # Both fields have to be good (True/1) for a goodVector
    goodField = np.logical_and(goodFieldPos, goodFieldNeg)
    print('Done')
    return goodField

def plot1DHist(x, numBins=50):
    # Display a simple histogram to see the approximate value spread
    hist, bins = np.histogram(x, bins=numBins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

def replaceH5Dataset(datasetName, fileHandle, newdata, saveAttrs=True, chunks=True, compression=None):

    datasetHandle = fileHandle[datasetName]
    if saveAttrs is True:
        existingAttributes = [(attr, value) for attr, value in datasetHandle.attrs.items()]

    del fileHandle[datasetName]
    fileHandle.create_dataset(datasetName, data=newdata,
                              chunks=chunks, compression=compression)
    if saveAttrs is True:
        for attr, value in existingAttributes:
            fileHandle[datasetName].attrs[attr] = value

picklePlace = 'AlkislarStats/'
# Perform simple example first on 1 set to get idea
[case1, case2, case3, case4, case5, case6, case7, case8, case9] = cases

########### Input ##########

#cases = [case4]
validateViaFlags =          True
validateViaStd =            True
stdCutOff =                 3.5
doLonePointInterp =         True
calcVorticity =             True
chunkSize =                 True #(50,50,50) # True just sets to auto-chunking. I can't figure out how to do a 3d chunk?

############################
cases = [case1, case4]
#cases = [case4]

for case in cases:
    hdf5Files = [PIVCasesInfo.setPath(setNumber) + 'SUBD/' + 'compiledNC.h5' for setNumber in case['sets']]
    #hdf5Files = hdf5Files[:1]
    print(hdf5Files)
    # sampleNC = currentFolder + 'subd00000008.nc'
    sampleChunks = h5py.File(hdf5Files[0], 'r')['axialVel'].chunks
    for h5File in hdf5Files:
        print('Starting:', h5File)
        flagFilterValidatedAttr = 'flagFilterValidated'
        # Load in total flag set
        fileHandle = h5py.File(h5File, 'r+')
        datasetHandle = fileHandle['flags']
        print('len of attrs:', len(datasetHandle.attrs))
        for attr, value in datasetHandle.attrs.items():
            print(attr, value)

        if validateViaFlags is True and flagFilterValidatedAttr not in datasetHandle.attrs:
            flagArray = np.array(datasetHandle[:])
            # First filter the images with the most number of interpolated flags.
            # Cut off fields with greater than 8% num interpolated flags.
            # ------------------------------------------------------------------
            # First filter the images with the most number of interpolated flags.
            GoodFields = np.ones(flagArray.shape[2], dtype=bool)
            percentCutOff = case['badImgCutOff']
            print('Checking for PIV fields with greater than', percentCutOff * 100, '% interpolated vectors...')

            for i in range(flagArray.shape[2]):
                # flags is a dictionary of unique values.
                # collections.Counter appears to only take 1 dimensional arrays.
                # flags is returned as a counter object. Use as you would a dictionary.
                #print('Field', i)
                flags = collections.Counter(flagArray[:, :, i].ravel())
                # flags: 1 = valid data, 10 = masked area, 33 = interpolated, 8 = ??? perhaps unrealistic magnitude and rejected.
                if flags[33] > percentCutOff * flagArray[:, :, i].shape[0] * flagArray[:, :, i].shape[1]:
                    print('Field:', i, 'has greater than ', percentCutOff * 100, '% interpolated vectors. Marked for removal.')
                    GoodFields[i] = False
            print('Total number of fields marked for removal:', np.sum(~GoodFields))

            # Now write on the datasets for keeps only if bad frames > 1
            if np.sum(~GoodFields) > 0:
                flagArray = flagArray[:, :, GoodFields]

                print('Now writing', 'flags', 'in place...')
                replaceH5Dataset(datasetName='flags', fileHandle=fileHandle, saveAttrs=True,
                                 newdata=flagArray.astype(np.int8),
                                 chunks=chunkSize)
                print('len of attrs:', len(datasetHandle.attrs))

                for dataset in ['axialVel', 'transverseVel']:
                    print('Loading in', dataset)
                    vectorHandle = fileHandle[dataset]
                    vectorArray = np.array(vectorHandle[:, :, GoodFields])

                    print('len of attrs:', len(vectorHandle.attrs))
                    vectorHandle.attrs[flagFilterValidatedAttr] = True
                    print('len of attrs:', len(vectorHandle.attrs))

                    print('Now writing', dataset, 'in place...')
                    replaceH5Dataset(datasetName=dataset, fileHandle=fileHandle, saveAttrs=True,
                                     newdata=vectorArray.astype(np.float32),
                                     chunks=chunkSize)
                del vectorArray

            datasetHandle.attrs[flagFilterValidatedAttr] = True
            fileHandle.flush()
            del flagArray

        else:
            print(h5File, 'has already been validated via flag numbers.')
            print('Skipping...')
        fileHandle.close()

        if validateViaStd is True:
            datasets = ['axialVel', 'transverseVel']
            for dataset in datasets:
                fileHandle = h5py.File(h5File, 'r+')
                datasetHandle = fileHandle[dataset]

                #Attributes to ensure set doesn't get validated several times.
                alreadyValidatedAttr = 'setStdValidated'
                badAreasRemovedAttr = 'badAreasRemoved'

                if alreadyValidatedAttr not in datasetHandle.attrs and datasetHandle.size > 0:
                    vectorArray = datasetHandle[:]

                    # Load in the saved fields for reference.
                    h5SavedStatsPath = picklePlace + 'case' + str(case['case']) + 'StatStore.h5'
                    h5SavedStats = h5py.File(h5SavedStatsPath, 'r')
                    meanLocation = 'casewise/mean/'+dataset
                    temporalMean = np.array(h5SavedStats[meanLocation])

                    stdLocation = 'casewise/std/'+dataset
                    temporalStd = np.array(h5SavedStats[stdLocation])

                    print('Starting h5file:', h5File, 'Dataset:', dataset)

                    # Remove the bad areas marked in the case dict.
                    if 'badAreas' in case:
                        # if any of these get tripped then we have done some process and need to write
                        # out the array again.
                        writeOnDataset = True
                        for ((row1, col1), (row2, col2)) in case['badAreas']:
                            print('Setting bad area to np.nan:', ((row1, col1), (row2, col2)))
                            vectorArray[col1:col2, row1:row2, :] = np.nan
                        datasetHandle.attrs[badAreasRemovedAttr] = True

                    print('Starting validation via standard deviation...')
                    goodVecs = validateStd(arr=vectorArray, temporalStd=temporalStd,
                                           temporalMean=temporalMean,
                                           stdCutOff=stdCutOff)
                    # Set the bad values to np.nan
                    beforeNans = np.count_nonzero(np.isnan(vectorArray))
                    #checkPoint = (229, 550)
                    checkPoint = (229, 750)
                    #plt.plot(vectorArray[checkPoint[0], checkPoint[1], :], ls='None', marker='.', markersize=5, color='r')
                    vectorArray[~goodVecs] = np.nan
                    #plt.plot(vectorArray[checkPoint[0], checkPoint[1], :], ls='None', marker='.', markersize=5, color='b')
                    #plt.show()
                    #exit(0)
                    afterNans = np.count_nonzero(np.isnan(vectorArray))
                    totalNans = (afterNans / vectorArray.size) * 100

                    print('Validation complete.')
                    print('Nans before:', beforeNans, 'Nans after:', afterNans, 'forming', totalNans,
                          '% of total entries')
                    datasetHandle.attrs[alreadyValidatedAttr] = True
                    #plt.contourf(vectorArray[:,:,20],30)

                    if doLonePointInterp is True:
                        # detect nans that are isolated and by themselves.
                        print('Starting interpolation of lone bad vectors.')
                        for i in range(vectorArray.shape[2]):
                            # Convolve filter with ones will produce value where nans are isolated on all edges.
                            currentField = vectorArray[:, :, i]
                            nanLocations = np.isnan(currentField)
                            counts = scipy.signal.convolve2d(nanLocations, np.ones((3, 3)), mode='same')
                            # Set the counts true to only where counts are equal to 1 or 2.
                            # This means that the nan is either
                            # by itself or has at most 1 nan neighbour.
                            validCounts = np.logical_or(counts == 1, counts == 2, counts == 3)
                            # The counts must be 1 to indicate that the only nan touching is itself.
                            fixableNans = np.logical_and(validCounts, nanLocations)

                            # Do griddata interpolation on the fixableNans
                            x = np.arange(currentField.shape[0])
                            y = np.arange(currentField.shape[1])
                            xx, yy = np.meshgrid(x, y)
                            xValid, yValid = np.where(~fixableNans)
                            xNans, yNans = np.where(fixableNans)

                            # Weighted Averaging via convolution -------------
                            # Taken from
                            #http: // stackoverflow.com / questions / 30068271 / python -
                            # get - get - average - of - neighbours - in -matrix -
                            #with-na - value
                            # print out only every 5th field to console to speed up


                            arr_pad = np.lib.pad(currentField, (1, 1), 'wrap')
                            R, C = xNans, yNans
                            N = arr_pad.shape[1]  # Number of rows in input array
                            offset = np.array([-N, -1, 1, N])
                            idx = np.ravel_multi_index((R + 1, C + 1), arr_pad.shape)[:, None] + offset
                            #currentField[R, C] = arr_pad.ravel()[idx].sum(1) / 4
                            # Taking the average which is the sum of the kernal values,
                            # divided by the number of valid points
                            # Counts start at 1, which includes the nan that the point is sitting on.
                            currentField[R, C] = np.nansum(arr_pad.ravel()[idx], axis=1) / (5 - counts[R, C])
                            if False:
                                plt.contourf(currentField, 50, cmap='magma')
                                plt.colorbar()
                                plt.plot(yNans, xNans, 'go', markersize=5)
                                #plt.plot(yNans, xNans, 'go', markersize=5)
                                plt.show()
                            if i % 10 == 0:
                                print('Interpolating field:', i, 'Interpolatable vectors:', np.sum(fixableNans),
                                      'or', '{:.3%}'.format(np.sum(fixableNans)/currentField.size))
                                totalNans = np.sum(np.isnan(currentField))
                                print('Nans for this field:', np.sum(np.isnan(currentField)), '. Percentage:',
                                      totalNans/currentField.size *100)
                                if False:
                                    plt.contourf(currentField, 50, cmap='magma')
                                    plt.colorbar()
                                    plt.plot(yNans, xNans, 'go', markersize=5)
                                    # plt.plot(yNans, xNans, 'go', markersize=5)
                                    plt.show()
                                    #exit(0)
                                    #datasetHandle.attrs['InterpolatedLoneNans'] = True

                    # delete the original dataset
                    # push the new dataset into that handle
                    print('Now writing', dataset, 'in place...')

                    replaceH5Dataset(datasetName=dataset, fileHandle=fileHandle, saveAttrs=True,
                                     newdata=vectorArray.astype(np.float32),
                                     chunks=chunkSize)
                    fileHandle.flush()
                    del vectorArray

                else:
                    # If the set has already been validated - do not validate twice!
                    print(h5File, 'has already been validated via standard deviation detection.')
                    print('Skipping...')

                fileHandle.close()


        print('Done.')
        #exit(0)
        if calcVorticity is True:
            h5Handle = h5py.File(h5File, 'r+')
            axialVelArrayHandle = h5Handle['axialVel']
            transverseVelArrayHandle = h5Handle['transverseVel']
            # Negative in premtive left right flip.
            axialVel = -np.fliplr(np.array(axialVelArrayHandle[:], dtype=np.float32))
            transverseVel = np.fliplr(np.array(transverseVelArrayHandle[:], dtype=np.float32))

            # dv/dx - du/dy
            print('Calculating vorticity...')
            dvdx = np.gradient(axialVel, edge_order=2, axis=0)
            del axialVel
            # Delete prematurely to save space.
            dudy = np.gradient(transverseVel, edge_order=2, axis=1)
            del transverseVel
            # trick here to save creation of third array
            vorticity = dvdx
            vorticity -= dudy
            # flip vorticity back
            vorticity = np.fliplr(vorticity)

            del dvdx, dudy
            print('Done.')
            # Write to dataset
            print('Writing vorticity...')
            if 'vorticity' in h5Handle:
                del h5Handle['vorticity']
            h5Handle.create_dataset('vorticity', data=vorticity.astype(np.float32), chunks=True,
                                    compression=None)
            print('Done.')
        print('Complete h5', h5File)
        # Blank line.
        print('')
