# Statistics again but using dask for efficient computation
# Written by Graham Bell 16/05/2016

import sys
import h5py
#try:
#    import h5py
#except:
#    sys.path.append("../h5py-master/")
#    import h5py
import numpy as np
#Packages should be installed but Massive is shit.
try:
    import dask.array as da
    import dask.array.stats
except:
    sys.path.append("../dask/")
    import dask.array as da

if (sys.version_info > (3, 0)):
    print('Imported python3 pickle')
    import pickle
    onMassive = False

else:
    # Python 2 code in this block
    print('Imported python2 cPickle')
    import cPickle as pickle
    print('onMassive Switch set to True.')
    onMassive = True

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#import compileNC
import PIVCasesInfo
import ideallyExpandedTools
import datetime
import scipy
import os
import itertools
import scipy.stats
from functools import partial
import multiprocessing
import time
#import collections
#sys.path.append("../PlotTools/")
# Custom plotting tools
import latexPlotTools as lpt
import pandas as pd

# Use the logging module.
if False:
    import logging

    level = logging.INFO
    format = '  %(message)s'
    logFile = 'daskPIVStats.log'
    try:
        os.remove(logFile)
    except:
        pass
    handlers = [logging.FileHandler(logFile), logging.StreamHandler()]
    logging.basicConfig(level = level, format = format, handlers = handlers)

#logging.info('Hey, this is working!')
import sys
logFile = 'daskPIVStats.log'
# Delete the old log file.
try:
    os.remove(logFile)
except:
    pass
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(logFile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()


def getStdContourLims(field, numStd):
    mean = np.nanmean(field)
    std = np.nanstd(field)
    maxLimit = mean + numStd*std
    minLimit = mean - numStd*std
    return minLimit, maxLimit

def chunks(longThing, chunkLen):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(longThing), chunkLen):
        yield longThing[i:i+chunkLen]

def multiproc_spearman(dataSubset, pointData):
    print(dataSubset.shape)
    correlation = np.zeros((dataSubset.shape[0], dataSubset.shape[1]))
    for (row,col), val in np.ndenumerate(dataSubset[:,:,0]):
        correlation[row,col],pval = scipy.stats.spearmanr(dataSubset[row,col], pointData)
    return correlation

def rowColToIndex(inputRowCol, arrayShape):
    return inputRowCol[0]*arrayShape[1] + inputRowCol[1]

def indexToRowCol(index, arrayShape):
    row, col = divmod(index, arrayShape[1])
    return row, col

def runOperationAndSave(userDict):
    operation = userDict['operation']
    datasets = userDict['datasets'] # change to iterable?
    print('################################')
    print('Starting', operation.__name__, 'at', datetime.datetime.now())
    for index, dataset in enumerate(datasets):
        print('')
        print('Starting dataset', dataset)
        start = time.time()
        userDict['datasetCurrentIndex'] = index

        temporalResult = operation(dataset=dataset, **userDict)
        end = time.time()
        print('Dataset completed in', end - start,'seconds')

        # Two point corr does the saving itself, this is only needed for the field functions.
        if 'writeQuantity' in userDict:
            # Chuck the dataset into the h5 file
            h5WriteLocation = userDict['writeGroup']+'/'+userDict['writeQuantity']+'/'+dataset
            #dict['h5SaveFile'][h5WriteLocation] = temporalResult
            # Overwrite the dataset if it exists
            print('Writing new dataset...')
            if h5WriteLocation in userDict['h5SaveFile']:
                print('Dataset already exists, deleting...')
                del userDict['h5SaveFile'][h5WriteLocation]
            dset = userDict['h5SaveFile'].create_dataset(name=h5WriteLocation, data=temporalResult,
                                                     shape=temporalResult.shape, dtype='float32')
            print('Done.')
            userDict['h5SaveFile'].flush()

    print('')
    print('Completed', operation.__name__, 'at', datetime.datetime.now())
    print('################################')
    print('')
    return

def daskReadHDF5Sets(hdf5FileNames, dataset, chunksize=(400,400,400), concat=True):
    data = [da.from_array(h5py.File(file, mode='r')[dataset], chunks=chunksize) for file in hdf5FileNames]
    if concat is True:
        data = da.concatenate(data, axis=2)
    print('Loaded dataset:', dataset)
    print('Dataset shape:', data.shape)
    print('Dataset size:', data.nbytes/1e9, 'GB')
    return data

def daskMean(dataset, h5DataFileNames, daskChunkSize, **kwargs):
    data = daskReadHDF5Sets(hdf5FileNames=h5DataFileNames, dataset=dataset, chunksize=daskChunksize)
    temporalMean = da.nanmean(data, axis=2, dtype='float64').compute()
    return np.array(temporalMean)

def daskStd(dataset, h5DataFileNames, daskChunkSize, **kwargs):
    data = daskReadHDF5Sets(hdf5FileNames=h5DataFileNames, dataset=dataset, chunksize=daskChunksize)
    temporalStd = da.nanstd(data, axis=2, dtype='float64').compute()
    return np.array(temporalStd)

def daskRMS(dataset, h5DataFileNames, daskChunkSize, **kwargs):
    data = daskReadHDF5Sets(hdf5FileNames=h5DataFileNames, dataset=dataset, chunksize=daskChunksize)
    square = data**2
    squareMean = da.nanmean(square, axis=2, dtype='float64').compute()
    rootMeanSquare = np.sqrt(squareMean)
    return rootMeanSquare

def daskSkewness(dataset, h5DataFileNames, daskChunkSize, **kwargs):
    # Skewness is caculated from mean and standard deviation data.
    # Load the mean data (required)
    print("Starting", dataset)
    data = daskReadHDF5Sets(hdf5FileNames=h5DataFileNames, dataset=dataset, chunksize=daskChunksize)

    # Stopped using pickles as it was giving me accuracy errors.
    mean = da.nanmean(data, axis=2, dtype='float64')
    std = da.nanstd(data, axis=2, dtype='float64')

    ExToThe3 = da.nanmean(data ** 3, axis=2, dtype='float64')
    # Use the computed mean to save computation time.
    temporalSkewness = (ExToThe3 - 3 * mean * std ** 2 - mean ** 3) / std ** 3
    temporalSkewness = temporalSkewness.compute()
    print("Completed", dataset)
    return np.array(temporalSkewness)

def daskKurtosis(dataset, h5DataFileNames, daskChunkSize, **kwargs):
    # Kurtosis is calculated from mean and standard deviation data.
    # Load the mean data (required)
    print("Starting", dataset)
    data = daskReadHDF5Sets(hdf5FileNames=h5DataFileNames, dataset=dataset, chunksize=daskChunksize)
    # Stopped using pickles as it was giving me accuracy errors.
    mean = da.nanmean(data, axis=2, dtype='float64')
    std = da.nanstd(data, axis=2, dtype='float64')

    # Reshape to 3d array to subtract using np broadcasting rules.
    #mean = mean.reshape((mean.shape[0], mean.shape[1], 1))
    # m4 = (((data - pickledMean)**4).mean(axis=2)).compute()
    # New method to ignore the np.nan from validation
    m4 = da.nanmean(((data - mean[:, :, np.newaxis]) ** 4), axis=2, dtype='float64')

    # Use the computed mean to save computation time.
    temporalKurtosis = (m4 / (std ** 4)).compute()
    return np.array(temporalKurtosis)

def dask2PointCorr(X, Y, Xbar=None, Ybar=None, stdX=None, stdY=None, temporalAxis=2):
    if Xbar is None:
        Xbar = da.nanmean(X, axis=2, dtype='float64')[:, :, np.newaxis]
        #Xbar = np.nanmean(X, axis=2, dtype='float64')[:, :, np.newaxis]
    if stdX is None:
        stdX = da.nanstd(X, axis=2, dtype='float64')
        #stdX = np.nanstd(X, axis=2, dtype='float64')
    if Ybar is None:
        Ybar = da.nanmean(Y, dtype='float64')
        #Ybar = np.nanmean(Y, dtype='float64')
    if stdY is None:
        stdY = da.nanstd(Y, dtype='float64')
        #stdY = np.nanstd(Y, dtype='float64')
    XPrime = X - Xbar
    YPrime = Y - Ybar
    n = da.sum(~da.isnan(X), axis=temporalAxis, dtype='float64')

    #n = np.sum(~np.isnan(X), axis=temporalAxis, dtype='float64')
    nMinus1 = n-1 #np.clip(n-1,a_min=0, a_max=np.max(n-1))
    if False:
        # Correltaion of the fluctuations?
        X = dataField0 - meanField0[:,:,np.newaxis]
        Xbar = da.nanmean(X,axis=2)
        XPrime = X - Xbar[:,:,np.newaxis]
        Y = refPointData - meanField0[field0RefPoint[1], field0RefPoint[0]]
        Ybar = da.nanmean(Y)
        YPrime = Y - Ybar
    covXY = (da.nansum(XPrime*YPrime, axis=temporalAxis, dtype='float64')/(nMinus1))
    #covXY = (np.nansum(XPrime * YPrime, axis=temporalAxis, dtype='float64') / (nMinus1))
    # Couldn't get this method to work due to n being different in X to Y due to np.nans
    #covXY2 = (da.nansum(X*Y, axis=2, dtype='float64') - (n*Xbar*Ybar))/ (nMinus1)
    corrXY = covXY/(stdX*stdY)
    print('Starting compute')
    corrXY = corrXY.compute()
    print('Done')
    # Old way of calculating it. Legacy code incase the parallelization methods are useful in the future.
    if False:
        for dataset, pickledMean, pickledStd in zip(datasets, [meanU, meanV, meanFlags], [stdU, stdV, stdFlags]):
            # (Row, Col)
            selectPoint = (700,310)

            data = [da.from_array(h5py.File(file,'r')[dataset], chunks=chunksize) for file in hdf5Files[:1]]
            data = da.concatenate(data, axis = 2)
            # Just concatenate the arrays together and return a numpy array for this example.
            print("Starting",dataset)
            #data = h5py.File(hdf5Files[0],'r')[dataset]
            print(data.shape)
            selectPointData = data[selectPoint[1],selectPoint[0],:].compute()
            selectPointData = selectPointData.flatten()

            # Put data in format: rows = observations, cols = variables
            #dataflat = data.transpose(2,0,1).reshape((data.shape[2],-1))
            # 20 by 20 chunks of the data
            numOfRowChunks = 100
            chunkRows = list(chunks(np.arange(data.shape[0]), numOfRowChunks))
            # Just get the first and last element for easier slicing for dask.
            chunkRows = [[chunk[0], chunk[-1]] for chunk in chunkRows]
            numOfColChunks = 100
            chunkCols = list(chunks(np.arange(data.shape[1]), numOfColChunks))
            # Just get the first and last element for easier slicing for dask.
            chunkCols = [[chunk[0], chunk[-1]] for chunk in chunkCols]

            chunkCoords = [(row,col) for (row,col) in itertools.product(chunkRows, chunkCols)]
            # Break this list up into groups of 7
            segmentChunkCoords = list(chunks(chunkCoords, 7))
            # Initialise a list
            spearcorr = []
            for chunkList in segmentChunkCoords:
                chunksPost = []
                for (rows, cols) in chunkList:
                    print(rows, cols)
                    #chunks.append(h5py.File(hdf5Files[0],'r')[dataset][row[0]:row[1],col[0]:col[1],:])
                    chunksPost.append(data[rows[0]:rows[1],cols[0]:cols[1],:].compute())
                    # Chunks must go to numpy arrays I cannot find a way around this.
                    # Therefore we will load in number of processor chunks. Do processing on that.
                    # Then load in next chunks
                func = partial(multiproc_spearman, pointData = selectPointData)
                # Set up for parallel processing; use 7 processors.
                p = multiprocessing.Pool(7)
                print('doing processing')
                tempResult = p.map(func, chunksPost)
                spearcorr.append(tempResult)

                p.close()
                p.join() # Wait unit all processes are done before continuing
                print('test')

            # Reconstuct the spearman correlation
            result = np.zeros((data.shape[0], data.shape[1]))
            # Recombine the array.
            for (row,col), corrArray in zip(chunkCoords, spearcorr):
                print(row, col, type(corrArray))
                result[row[0]:row[1],col[0]:col[1]] = corrArray
            plt.contourf(result, 30, cmap = 'viridis')
            plt.show()

            exit(0)
            '''
            data = [da.from_array(h5py.File(file,'r')[dataset], chunks=(400,400,500)) for file in hdf5Files]
            data = da.concatenate(data, axis = 2)
            print("Starting",dataset)
            # Select point is the index of the flipped matrix
            #( ROW, COL)
            selectPoint = (700,310)
            selectDataMean = data[selectPoint[1],selectPoint[0],:] - pickledMean[selectPoint[1],selectPoint[0]]
            selectDataMean = selectDataMean.reshape((1,1,data.shape[2]))
            pickledMean = pickledMean.reshape((pickledMean.shape[0], pickledMean.shape[1], 1))
            # Useful for seeing where the spot is.
            #plt.contourf(pickledMean, 30)
            #plt.plot(selectPoint[0], selectPoint[1], 'bo')
            #plt.show()
            #exit(0)

            import time
            start = time.time()
            TwoPtCorr = da.mean((data - pickledMean)*selectDataMean, axis=2)   #*(selectDataMean))  #da.mean(selectData*data, axis = 2) - (pickledMean[selectPoint[1],selectPoint[0]] * pickledMean)
            TwoPtCorr = TwoPtCorr.compute()
            TwoPtCorr = TwoPtCorr/(pickledStd[selectPoint[1],selectPoint[0]]*pickledStd)

            end = time.time()

            print("Completed",dataset,'in time:',end - start)
            '''

            #dataToPickle.append(TwoPtCorr)
    return np.array(corrXY)

def spatialCorrManager(dataset, datasetIndexes, datasetCurrentIndex, h5DataFileNames, daskChunkSize,
                       refPointDict, h5CorrOutFileName, **kwargs):

    # dataset is actually passed as a tuple pair
    # datasetIndex is actually passed as a tuple pair for two point correlation.
    datasetTitle0, datasetTitle1 = dataset
    datasetIndex0, datasetIndex1 = datasetIndexes[datasetCurrentIndex]

    print('Computing correlation along reference lines:', refPointDict.keys())

    # Can't get dask to run quick enough, pull as numpy array.
    dataField0 = np.array(daskReadHDF5Sets(hdf5FileNames=h5DataFileNames, dataset=datasetTitle0,
                                           chunksize=daskChunkSize))
    # only need one row col from the dataField1 case. Don't turn that into an array just yet.
    # This avoids loading in the whole dataset.
    dataField1 = daskReadHDF5Sets(hdf5FileNames=h5DataFileNames, dataset=datasetTitle1,
                                  chunksize=daskChunkSize)
    #load it into memory then treat it again as a dask computation.
    #dataField = da.from_array(dataField, chunks=chunksize)

    for refPointParent, refPointSet in refPointDict.items():
        print('Starting ref point parent:', refPointParent)
        for refPointindex, refPoint in enumerate(refPointSet):
            print("Starting", dataset, ', ref point', refPoint,':', refPointindex+1, 'of', len(refPointSet),
                  'at', datetime.datetime.now())

            h5File = h5py.File(h5CorrOutFileName, mode='a')
            # open the h5 file.

            # Create eg the 'axialVel' group/folder
            # datasetGroup = h5File.create_group(dataset)
            # Within that folder stick the index style correlation point fields.
            # Write an index to (row, col) converter.

            flatIndex = rowColToIndex(inputRowCol=refPoint, arrayShape=dataField0.shape[:2])
            print('Ref point', refPoint, 'corresponds to flat index', flatIndex)

            # Use the combined group and node format for addressing
            #dSetFilePath = dataset + '/' + str(flatIndex)
            corrNameAlt0 = 'R'+str(datasetIndex0) + str(datasetIndex1)
            corrNameAlt1 = 'R' + str(datasetIndex1) + str(datasetIndex0)
            dSetFilePathAlt0 = corrNameAlt0 + '/' + str(flatIndex)
            dSetFilePathAlt1 = corrNameAlt1 + '/' + str(flatIndex)
            # Don't overwrite the points anymore
            if False:
                if dSetFilePath in h5File:
                    dset = h5File[dSetFilePath]
                    for attribute in dset.attrs:
                        print(attribute)
            if False:
                if dSetFilePath in h5File:
                    dset = h5File[dSetFilePath]
                    for attribute in dset.attrs:
                        if 'parent' in attribute:
                            del dset.attrs[attribute]
                        #print(attribute)

            if dSetFilePathAlt0 not in h5File and dSetFilePathAlt1 not in h5File:
                start = time.time()
                refRow = refPoint[0]
                refCol = refPoint[1]

                # Load the ref point data into memory.
                refPointData = np.array(dataField1[refRow, refCol, :])
                # recast as in memory da array
                refPointDataDaskd = da.from_array(refPointData, chunks=refPointData.shape)

                X = da.from_array(dataField0, chunks=daskChunkSize)
                Y = refPointDataDaskd
                corrXY = dask2PointCorr(X=X, Y=Y)#, Xbar=Xbar, Ybar=Ybar, stdX=stdX, stdY=stdY, temporalAxis=2)

                print('before end')
                end = time.time()
                print('Dataset completed in', end - start,'seconds')
                print('Writing correlation result to file', h5CorrOutFileName)
                # Overwrite the dataset if it exists
                print('Writing new dataset...')
                dset = h5File.create_dataset(name=dSetFilePathAlt0, data=corrXY, shape=corrXY.shape,
                                                 dtype='float32')
                dset.attrs['numSamples'] = dataField0.shape[2]

                dset = h5File[dSetFilePathAlt0]
                dset.attrs['parent_' + refPointParent] = refPointParent
                print('Pushing parent', 'parent_' + refPointParent)
                print('Done.')
                h5File.flush()
            else:
                print('Ref point', refPoint, 'already in file.')
                print('Skipping...')
    h5File.close()
    print('')
    return

def daskTemporalPDF(dataset, datasetIndex, hdf5FileNames, chunksize, refPoints, **kwargs):

    meanField = pickle.load(open(pickleFileMean, "rb"))[datasetIndex]
    for refPoint in refPoints:
        fig = plt.figure()
        axL = fig.add_subplot(1, 2, 1)
        axR = fig.add_subplot(1, 2, 2)
        # Do the compiled cases.
        if False:
            data = daskReadHDF5Sets(hdf5FileNames, dataset, chunksize)
            # Do the PDF
            print('Taking on refPoint:', refPoint)
            refRow, refCol = refPoint
            print('Dragging data from file...')
            spotData = np.array(data[refRow, refCol, :])
            print('Done.')
            spotData = spotData[~np.isnan(spotData)]
            #hist, bin_edges = np.histogram(spotData, bins=60)
            #axR.plot(bin_edges[:-1], hist)
            axR.plot(spotData, ls='None', marker='.', markersize=0.3)
            extension = '_trace.png'
        if True:
            data = daskReadHDF5Sets(hdf5FileNames, dataset, chunksize)
            # Do the PDF
            print('Taking on refPoint:', refPoint)
            refRow, refCol = refPoint
            print('Dragging data from file...')
            spotData = np.array(data[refRow, refCol, :])
            print('Done.')
            spotData = spotData[~np.isnan(spotData)]
            spotData = np.clip(a=spotData, a_min=-25, a_max=-20)
            hist, bin_edges = np.histogram(spotData, bins=100)
            axR.plot(bin_edges[:-1], hist)
            skewnessValue = scipy.stats.skew(spotData, nan_policy='omit')
            axR.set_title('Skewness:'+str(skewnessValue))
            #axR.plot(spotData, ls='None', marker='.', markersize=0.3)
            extension = '_hist.png'

        if False:
            for hdf5FileName in hdf5FileNames:
                print('hdf5 name', hdf5FileName)
                data = daskReadHDF5Sets([hdf5FileName], dataset, chunksize)
                # Do the PDF
                print('Taking on refPoint:', refPoint)
                refRow, refCol = refPoint
                print('Dragging data from file...')
                spotData = np.array(data[refRow, refCol, :])
                print('Done.')
                spotData = spotData[~np.isnan(spotData)]
                hist, bin_edges = np.histogram(spotData, bins=60)
                axR.plot(bin_edges[:-1], hist, label=hdf5FileName)

            extension = '_hist.png'

        axL.contourf(meanField, 30, cmap='inferno')
        axL.plot(refCol, refRow, 'ro')
        axL.set_aspect('equal')
        axR.set_xlabel('Displacement')
        axR.set_ylabel('Count')
        #axR.legend()
        plotPath = './temporalPDFs/'+dataset+'/'
        os.makedirs(plotPath, exist_ok=True)
        plotName = suffixFilename + '_pointRow' + str(refRow) + '_Col' + str(refCol)
        plt.savefig(plotPath+plotName+extension, dpi=200)
        plt.close(fig)
    return None

def daskSaveInstantaneous(dataset, h5DataFileNames, daskChunkSize, **kwargs):
    data = daskReadHDF5Sets(hdf5FileNames=h5DataFileNames, dataset=dataset, chunksize=daskChunksize)

    textWidth = 489.0  # points
    textWidth_inches = textWidth / 72.0

    width = textWidth_inches * 0.7
    fig = plt.figure(figsize=(width, 0.7 * width))

    ax = fig.add_subplot(1, 1, 1)
    snapshot = np.fliplr(-data[:, :, 7100])

    if True:
        # Linearly interpolate out the bad vectors.
        from scipy import interpolate
        x = np.arange(0, snapshot.shape[1])
        y = np.arange(0, snapshot.shape[0])
        array = np.ma.masked_invalid(snapshot)
        xx, yy = np.meshgrid(x, y)
        # get only the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]

        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                   (xx, yy),
                                   method='cubic')
        snapshot = GD1

    # Scale the snapshot to u/u_j

    frameTime = case['dt'] * 1e-6  # in s:
    snapshot = case['mag'] * snapshot / frameTime

    # Non-dimentionalise to the reference the ideally expanded velocity Uj

    Ue = ideallyExpandedTools.throatVelocity(T0=(15 + 273))
    snapshot /= Ue

    goodCbarLims = lpt.goodCbarLims(array=snapshot, cbarStd=2)

    xAxisCoords = case['pivPixelCoordsCols']
    yAxisCoords = case['pivPixelCoordsRows']
    cont = ax.pcolormesh(xAxisCoords, yAxisCoords, snapshot, vmin=0, vmax=2.0, cmap='magma')
    ax.set_xlim(left=0)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x/D$')
    ax.set_ylabel(r'$y/D$')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable=cont, cax=cax)
    cbar.set_ticks([0,1,2])
    cbar.set_label(r'$u_x/U_{e}$')

    fig.tight_layout(pad=0.1)


    saveName = 'case4_instantaneousPIVSnapshot.png'
    fig.savefig(fname=saveName, dpi=300)
    exit(0)
    return

def resample3DArray(indexes):
    # This function is operated on by the individual processes
    return sharedArray[:, :, indexes]

def parallelResampleManager(array, indexes):
    # This function coordinates the parallelization.
    # break the indexes into chunks
    numProcess = multiprocessing.cpu_count()
    # Apparently multiprocess can't pass enormous chunks back and forth. So we'll limit the chunks it passes back
    # by adding more jobs.
    if len(indexes) < 1000:
        indexChunks = np.array_split(ary=indexes, indices_or_sections=numProcess)
    else:
        indexChunks = chunks(longThing=indexes, chunkLen=300)
    # zip the index chunks and array together to pass to functions.
    # import sharedmem
    # sharedData = sharedmem.copy(array)
    # iterable = zip(itertools.repeat(sharedData, times=len(indexChunks)), indexChunks)
    # Globally access the shared array. Shit solution I know, wasn't able to get the sharedmem solution to work.
    global sharedArray
    sharedArray = array
    pool = multiprocessing.Pool(numProcess)
    # resampledArray = pool.map(func=resample3DArray, iterable=indexChunks)
    resampledArray = pool.map_async(func=resample3DArray, iterable=indexChunks).get()

    # resampledArray = pool.starmap(func=resample3DArray, iterable=iterable)
    pool.close()
    pool.join()  # Wait unit all processes are done before continuing
    del sharedArray
    resampledArray = np.dstack(resampledArray)
    return resampledArray

def daskBootstrap(dataset, h5DataFileNames, daskChunkSize, numResamples, h5FileNameOut, daskfunc=da.nanmean,
                  funcDescript='mean', **kwargs):
    # Passed back as a dask array.
    # Should be small enough to load into memory. ~ 30 GB for the axial velocity case only.
    print('Started Read-in...')
    data = np.array(daskReadHDF5Sets(hdf5FileNames=h5DataFileNames, dataset=dataset, chunksize=daskChunksize))
    dataAsDask = da.from_array(data, chunks=daskChunksize)
    print('Completed Read.')
    resampledFields = np.zeros(shape=(data.shape[0], data.shape[1], numResamples))
    for i in range(numResamples):
        print('Resample number:', i + 1, '/', numResamples)
        # Grab a random set of fields from the orig dataset.
        # Sort to make dask's life a bit easier.
        print('Doing reshuffle')
        randomIndexes = np.sort(np.random.randint(data.shape[2], size=data.shape[2]))
        # shuffledData = parallelResampleManager(array=data, indexes=randomIndexes)
        print('Reshuffle stored')
        # Try and work on this in parallel.
        # resampledData = da.from_array(shuffledData, chunks=daskChunksize)
        # print('Starting mean')
        print('Starting' , funcDescript)
        # resampledStat = daskfunc(resampledData, axis=2, dtype='float64').compute()
        resampledStat = daskfunc(dataAsDask[: ,: , randomIndexes], axis=2, dtype='float64').compute()
        print('Done', funcDescript)
        # Slot it in
        resampledFields[:, :, i] = resampledStat

    # Provide the mean and std fields of this as well.
    bootstrapResult = da.from_array(resampledFields, chunks=daskChunksize)
    bootStrapMean = da.nanmean(bootstrapResult, axis=2, dtype='float64').compute()
    bootStrapStd = da.nanstd(bootstrapResult, axis=2, dtype='float64').compute()

    h5File = h5py.File(h5FileNameOut, mode='w')
    print('Writing new dataset...')
    h5File.create_dataset(name=dataset+'/' + 'resampled_' + funcDescript,
                          data=resampledFields, shape=resampledFields.shape, dtype='float32')
    h5File.create_dataset(name=dataset+'/' + 'bootstrapMean', data=bootStrapMean, shape=bootStrapMean.shape, dtype='float32')
    h5File.create_dataset(name=dataset+'/' + 'bootstrapStd', data=bootStrapStd, shape=bootStrapStd.shape, dtype='float32')



    # open the h5 file.
    h5File.flush()
    h5File.close()
    print('Completed H5.')



def calcBootstrapConfInterval(sampleStatH5FileLoc, sampleStatLoc, bootStrapH5FileLoc, bootstrapStdLoc, **kwargs):

    sampleStatH5SaveFile = h5py.File(sampleStatH5FileLoc, mode='r')
    sampleStat = np.array(sampleStatH5SaveFile[sampleStatLoc])
    sampleStatH5SaveFile.close()
    # bootstrapMean = da.from_array(h5SaveFile[bootstrapStatLoc], chunks=daskChunksize)
    bootstrapH5SaveFile = h5py.File(bootStrapH5FileLoc, mode='r')

    bootstrapStd = np.array(bootstrapH5SaveFile[bootstrapStdLoc])
    bootstrapH5SaveFile.close()
    bootstrap95 = 1.96*bootstrapStd
    bootstrap95_norm = np.abs(bootstrap95/sampleStat)

    h5File = h5py.File(bootStrapH5FileLoc, mode='a')
    bootstrapNormDsetName = '/axialVel/' + 'bootstrap95_norm'
    if bootstrapNormDsetName in h5File:
        del h5File[bootstrapNormDsetName]
    h5File.create_dataset(name=bootstrapNormDsetName, data=bootstrap95_norm, shape=bootstrap95_norm.shape,
                          dtype='float32')
    # Joel's code.
    # rs_u_mean_mean = mean(rs_u_mean, 3);
    # rs_u_mean_var = var(rs_u_mean, 0, 3);
    # rs_u_mean_std = sqrt(rs_u_mean_var);
    # rs_u_mean_95 = 1.96. * rs_u_mean_std;
    #
    # rs_u_mean_95_norm = abs(rs_u_mean_95. / u_mean);
    return



def plotField():
    print("Loaded pickle:", pickleFileMean)
    [meanU, meanV, meanFlags] = pickle.load(open(pickleFileMean, "rb"))

    if True:
        ####For AFMC ############
        # Plot the u component, v component, and |uv| component.
        pickleFileMeanCase4 = picklePlace + 'case' + str(case4['case']) + 'Mean.dat'
        [meanUCase4, meanVCase4, meanFlagsCase4] = pickle.load(open(pickleFileMeanCase4, "rb"))
        #########################


    meanArrays = [-meanU, meanV, np.sqrt(meanU**2 + meanV**2)]
    meanFilenames = ['u', 'v', 'uv']
    meanTitles = [r'$\langle u/U_{e} \rangle$', r'$\langle v/U_{e} \rangle$', r'$\langle |uv|/U_{e} \rangle$']
    for meanArray, meanFilename, meanTitle in zip(meanArrays, meanFilenames, meanTitles):
        frameTime = case['dt']*1e-6 # in s:
        magUV = case['mag'] * meanArray/frameTime
        # Remove the mask
        magUV[magUV == 0.0] = np.nan
        # Non-dimentionalise to the reference the ideally expanded velocity Uj

        Ue = ideallyExpandedTools.convergingExitVelocity(flowTemp=(15+273))
        magUV = magUV/Ue
        magUV = np.fliplr(magUV)

        if True:
            #### For AFMC #####
            # Plot half of the 4.6 case on the top half of the image.
            meanUCase4 = case['mag'] * meanUCase4/frameTime
            meanUCase4 = meanUCase4/Ue
            meanUCase4 = np.fliplr(-meanUCase4)
            # Insert in upperhalf of data
            magUV[:case['zeroRow'],:] = meanUCase4[:case['zeroRow'],:]
            magUV[:,:case['zeroCol']] = np.nan
            ##########

        # Plot the max to the xth std of the field.
        #minContourLevel, maxContourLevel = getStdContourLims(magUV, 10.0)
        # Levels (values of the colorbar)
        numLevels = 30
        levels = np.linspace(np.nanmin(magUV), np.nanmax(magUV), numLevels, endpoint=True)
        # Where the labels will actually be placed
        numTicks = 9
        cbarTicks = np.linspace(np.nanmin(magUV), np.nanmax(magUV), numTicks, endpoint=True)
        #cbarTicks[1:-1] = np.round(cbarTicks[1:-1], decimals=2)
        # round the points inbetween the min and max

        cbarLabel = meanTitle
        plotTitle = 'Mean (' + meanTitle + ' component)' + suffixTitle
        plotFilename = picklePlace + 'mean/' + 'case' + str(case['case']) + 'Mean' + meanFilename + suffixFilename + '.png'
        fig = lpt.plotContour(magUV, plotTitle, levels, cbarLabel, cbarTicks, pivPixelCoordsRows=pivPixelCoordsRows, pivPixelCoordsCols=pivPixelCoordsCols)

        if True:
            ###### For AFMC ########
            for shockLoc in case4['shockReflectionPoints']:
                plt.axvline(pivPixelCoordsCols[shockLoc],ymin = 0, ymax = 0.5, color = 'r', lw = 0.7, alpha = 0.8)
            for shockLoc in case1['shockReflectionPoints']:
                plt.axvline(pivPixelCoordsCols[shockLoc],ymin = 0.5, ymax = 1, color = 'r', lw = 0.7, alpha = 0.8)
            # Centre divider
            plt.axhline(pivPixelCoordsRows[case['zeroRow']], color = 'w')
        else:
            for shockLoc in case['shockReflectionPoints']:
                plt.axvline(pivPixelCoordsCols[shockLoc],ymin = 0.0, ymax = 1, color = 'r', lw = 0.7, alpha = 0.8)

        plt.tight_layout()

        plt.savefig(plotFilename, dpi=dpi)
        plt.close()

def runDaskStats(case, picklePlace, operations, pickleFiles, datasets, chunksize=(400,400,400)):
    print('Starting case', case['case'], datetime.datetime.now())


    if do2PointCorr is True:
        dataToPickle = []
        #Load the mean data (required)
        print("Loaded pickle:", pickleFileMean)
        [meanU, meanV, meanFlags] = pickle.load(open(pickleFileMean, "rb"))
        # Load the standard deviation data (required)
        print("Loaded pickle:", pickleFileStd)
        [stdU, stdV, stdFlags] = pickle.load(open(pickleFileStd, "rb"))
        for dataset, pickledMean, pickledStd in zip(datasets, [meanU, meanV, meanFlags], [stdU, stdV, stdFlags]):
            # (Row, Col)
            selectPoint = (700,310)

            data = [da.from_array(h5py.File(file,'r')[dataset], chunks=chunksize) for file in hdf5Files[:1]]
            data = da.concatenate(data, axis = 2)
            # Just concatenate the arrays together and return a numpy array for this example.
            print("Starting",dataset)
            #data = h5py.File(hdf5Files[0],'r')[dataset]
            print(data.shape)
            selectPointData = data[selectPoint[1],selectPoint[0],:].compute()
            selectPointData = selectPointData.flatten()

            # Put data in format: rows = observations, cols = variables
            #dataflat = data.transpose(2,0,1).reshape((data.shape[2],-1))
            # 20 by 20 chunks of the data
            numOfRowChunks = 100
            chunkRows = list(chunks(np.arange(data.shape[0]), numOfRowChunks))
            # Just get the first and last element for easier slicing for dask.
            chunkRows = [[chunk[0], chunk[-1]] for chunk in chunkRows]
            numOfColChunks = 100
            chunkCols = list(chunks(np.arange(data.shape[1]), numOfColChunks))
            # Just get the first and last element for easier slicing for dask.
            chunkCols = [[chunk[0], chunk[-1]] for chunk in chunkCols]

            import itertools
            import scipy.stats
            from functools import partial
            import multiprocessing
            chunkCoords = [(row,col) for (row,col) in itertools.product(chunkRows, chunkCols)]
            # Break this list up into groups of 7
            segmentChunkCoords = list(chunks(chunkCoords, 7))
            # Initialise a list
            spearcorr = []
            for chunkList in segmentChunkCoords:
                chunksPost = []
                for (rows, cols) in chunkList:
                    print(rows, cols)
                    #chunks.append(h5py.File(hdf5Files[0],'r')[dataset][row[0]:row[1],col[0]:col[1],:])
                    chunksPost.append(data[rows[0]:rows[1],cols[0]:cols[1],:].compute())
                    # Chunks must go to numpy arrays I cannot find a way around this.
                    # Therefore we will load in number of processor chunks. Do processing on that. Then load in next chunks
                func = partial(multiproc_spearman, pointData = selectPointData)
                # Set up for parallel processing; use 7 processors.
                p = multiprocessing.Pool(7)
                print('doing processing')
                tempResult = p.map(func, chunksPost)
                spearcorr.append(tempResult)

                p.close()
                p.join() # Wait unit all processes are done before continuing
                print('test')

            # Reconstuct the spearman correlation
            result = np.zeros((data.shape[0], data.shape[1]))
            # Recombine the array.
            for (row,col), corrArray in zip(chunkCoords, spearcorr):
                print(row, col, type(corrArray))
                result[row[0]:row[1],col[0]:col[1]] = corrArray
            plt.contourf(result, 30, cmap = 'viridis')
            plt.show()

            exit(0)
            '''
            data = [da.from_array(h5py.File(file,'r')[dataset], chunks=(400,400,500)) for file in hdf5Files]
            data = da.concatenate(data, axis = 2)
            print("Starting",dataset)
            # Select point is the index of the flipped matrix
            #( ROW, COL)
            selectPoint = (700,310)
            selectDataMean = data[selectPoint[1],selectPoint[0],:] - pickledMean[selectPoint[1],selectPoint[0]]
            selectDataMean = selectDataMean.reshape((1,1,data.shape[2]))
            pickledMean = pickledMean.reshape((pickledMean.shape[0], pickledMean.shape[1], 1))
            # Useful for seeing where the spot is.
            #plt.contourf(pickledMean, 30)
            #plt.plot(selectPoint[0], selectPoint[1], 'bo')
            #plt.show()
            #exit(0)

            import time
            start = time.time()
            TwoPtCorr = da.mean((data - pickledMean)*selectDataMean, axis=2)   #*(selectDataMean))  #da.mean(selectData*data, axis = 2) - (pickledMean[selectPoint[1],selectPoint[0]] * pickledMean)
            TwoPtCorr = TwoPtCorr.compute()
            TwoPtCorr = TwoPtCorr/(pickledStd[selectPoint[1],selectPoint[0]]*pickledStd)

            end = time.time()

            print("Completed",dataset,'in time:',end - start)
            '''

            #dataToPickle.append(TwoPtCorr)
        #dataToPickle.append(selectPoint)
        #pickle.dump(dataToPickle, open(pickleFile2PtCorr, 'wb'))
        print("Written mean data to:",pickleFile2PtCorr)

    if plotSnapshot is True:

        data = da.from_array(h5py.File(hdf5Files[0],'r')[datasets[0]], chunks=(1500,1500,1))
        uVel = np.fliplr(data[:,:,20])
        data = da.from_array(h5py.File(hdf5Files[0],'r')[datasets[1]], chunks=(1500,1500,1))
        vVel = np.fliplr(data[:,:,20])
        magUV = uVel**2 + vVel**2

        frameTime = case['dt']*1e-6 # in s:
        magUV = case['mag'] * np.sqrt(magUV)/frameTime

        # Remove the mask from 0s to np.nans.
        magUV[magUV == 0.0] = np.nan
        # mask the region where there is no laser
        #magUV[:,:50] = np.nan

        calcNozzleExitVel = 310.5 #m/s
        magUV = magUV/calcNozzleExitVel

        # Levels (values of the colorbar)
        numLevels = 15
        levels = np.linspace(0.0, 2.0, 20, endpoint=True)
        # Where the labels will actually be placed
        numTicks = 9
        cbarTicks = np.linspace(0.0, 2.0, numTicks, endpoint=True)
        cbarTicks[1:-1] = np.round(cbarTicks[1:-1], decimals=2)
        # round the points inbetween the min and max

        snapFilename = 'uv'
        cbarLabel = r'$\sqrt{u_{x}^2 + u_{y}^2} / U_{j}$'
        plotTitle = r'$\mathrm{Velocity \, Magnitude}$' "\n" '$\mathrm{Nozzle \, Pressure \, Ratio: \, 4.6}$' #+ str(case['NPR'])
        plotFilename = picklePlace + 'snapShots/' + 'case' + str(case['case']) + 'snapShot' + snapFilename + suffixFilename + '.png'

        plotContour(magUV, plotFilename, plotTitle, levels, cbarLabel, cbarTicks)

    if plotVectorSnapshot is True:
        skip = 1
        data = da.from_array(h5py.File(hdf5Files[6],'r')[datasets[0]], chunks=(1500,1500,1))
        uVel = -np.fliplr(data[::skip,::skip,20])
        data = da.from_array(h5py.File(hdf5Files[6],'r')[datasets[1]], chunks=(1500,1500,1))
        vVel = np.fliplr(data[::skip,::skip,20])

        frameTime = case['dt']*1e-6 # in s:
        Ue = ideallyExpandedTools.convergingExitVelocity(flowTemp=(15 + 273))
        uVel = uVel
        vVel = vVel
        uVel = case['mag'] * uVel / frameTime
        vVel = case['mag'] * vVel / frameTime
        uvMag = np.sqrt(uVel**2 + vVel**2)/Ue
        # Display as a vector image
        numLevels = 50

        minContourLevel, maxContourLevel = getStdContourLims(uvMag, 2.0)
        levels = np.linspace(minContourLevel, maxContourLevel, numLevels, endpoint=True)
        # Where the labels will actually be placed
        numTicks = 9
        cbarTicks = np.linspace(minContourLevel, maxContourLevel, numTicks, endpoint=True)
        # cbarTicks[1:-1] = np.round(cbarTicks[1:-1], decimals=2)
        # round the points inbetween the min and max

        cbarLabel = r'$\vec{\mathbf{V}}/U_e$'
        plotTitle = 'Vector Sample ' + suffixTitle

        fig = lpt.plotContour(uvMag, plotTitle, levels, cbarLabel, cbarTicks, pivPixelCoordsRows=pivPixelCoordsRows,
                              pivPixelCoordsCols=pivPixelCoordsCols, figWidth=1.0)
        #plt.quiver(pivPixelCoordsCols, pivPixelCoordsRows, uVel, vVel, scale=500, width=0.0009, alpha=0.8)

        #plotFilename = picklePlace + 'snapshots/' + 'case' + str(case['case']) + 'Mean' + meanFilename + suffixFilename + '.png'
        snapFilename = 'uvMag'
        plotTitle = r'$\mathrm{Velocity \, Magnitude}$' "\n" '$\mathrm{Nozzle \, Pressure \, Ratio: \, 4.6}$' #+ str(case['NPR'])
        plotFilename = picklePlace + 'snapShots/' + 'case' + str(case['case']) + 'snapShotVector' + snapFilename + suffixFilename + '.png'
        plt.tight_layout()
        plt.savefig(plotFilename, dpi=dpi)
        plt.clf()
        # Now Zoomed Image

        rowMin = 170
        rowMax = 250
        colMin = 150
        colMax = 200

        cbarLabel = '\arrow{\mathbf{V}}'
        plotTitle = 'Vector Sample ' + suffixTitle

        fig = lpt.plotContour(uvMag[rowMin:rowMax, colMin:colMax], plotTitle, levels, cbarLabel, cbarTicks,
                              pivPixelCoordsRows=pivPixelCoordsRows[rowMin:rowMax],
                              pivPixelCoordsCols=pivPixelCoordsCols[colMin:colMax], figWidth=1.0)
        uVel = uVel/uvMag
        vVel = vVel/uvMag
        plt.quiver(pivPixelCoordsCols[colMin:colMax], pivPixelCoordsRows[rowMin:rowMax],
                   uVel[rowMin:rowMax, colMin:colMax], vVel[rowMin:rowMax, colMin:colMax], headaxislength=4, scale = 18000, width=0.0015) #, scale=1000, width=0.0009,
                   #alpha=0.8)

        snapFilename = 'uvMag'
        plotTitle = r'$\mathrm{Velocity \, Magnitude}$' "\n" '$\mathrm{Nozzle \, Pressure \, Ratio: \, 4.6}$'  # + str(case['NPR'])
        plotFilename = picklePlace + 'snapShots/' + 'case' + str(
            case['case']) + 'snapShotVectorZoom' + snapFilename + suffixFilename + '.png'
        plt.tight_layout()
        plt.savefig(plotFilename, dpi=dpi)
        exit(0)

    if plotMean is True:
        print("Loaded pickle:", pickleFileMean)
        [meanU, meanV, meanFlags] = pickle.load(open(pickleFileMean, "rb"))

        if True:
            ####For AFMC ############
            # Plot the u component, v component, and |uv| component.
            pickleFileMeanCase4 = picklePlace + 'case' + str(case4['case']) + 'Mean.dat'
            [meanUCase4, meanVCase4, meanFlagsCase4] = pickle.load(open(pickleFileMeanCase4, "rb"))
            #########################


        meanArrays = [-meanU, meanV, np.sqrt(meanU**2 + meanV**2)]
        meanFilenames = ['u', 'v', 'uv']
        meanTitles = [r'$\langle u/U_{e} \rangle$', r'$\langle v/U_{e} \rangle$', r'$\langle |uv|/U_{e} \rangle$']
        for meanArray, meanFilename, meanTitle in zip(meanArrays, meanFilenames, meanTitles):
            frameTime = case['dt']*1e-6 # in s:
            magUV = case['mag'] * meanArray/frameTime
            # Remove the mask
            magUV[magUV == 0.0] = np.nan
            # Non-dimentionalise to the reference the ideally expanded velocity Uj

            Ue = ideallyExpandedTools.convergingExitVelocity(flowTemp=(15+273))
            magUV = magUV/Ue
            magUV = np.fliplr(magUV)

            if True:
                #### For AFMC #####
                # Plot half of the 4.6 case on the top half of the image.
                meanUCase4 = case['mag'] * meanUCase4/frameTime
                meanUCase4 = meanUCase4/Ue
                meanUCase4 = np.fliplr(-meanUCase4)
                # Insert in upperhalf of data
                magUV[:case['zeroRow'],:] = meanUCase4[:case['zeroRow'],:]
                magUV[:,:case['zeroCol']] = np.nan
                ##########

            # Plot the max to the xth std of the field.
            #minContourLevel, maxContourLevel = getStdContourLims(magUV, 10.0)
            # Levels (values of the colorbar)
            numLevels = 30
            levels = np.linspace(np.nanmin(magUV), np.nanmax(magUV), numLevels, endpoint=True)
            # Where the labels will actually be placed
            numTicks = 9
            cbarTicks = np.linspace(np.nanmin(magUV), np.nanmax(magUV), numTicks, endpoint=True)
            #cbarTicks[1:-1] = np.round(cbarTicks[1:-1], decimals=2)
            # round the points inbetween the min and max

            cbarLabel = meanTitle
            plotTitle = 'Mean (' + meanTitle + ' component)' + suffixTitle
            plotFilename = picklePlace + 'mean/' + 'case' + str(case['case']) + 'Mean' + meanFilename + suffixFilename + '.png'
            fig = lpt.plotContour(magUV, plotTitle, levels, cbarLabel, cbarTicks, pivPixelCoordsRows=pivPixelCoordsRows, pivPixelCoordsCols=pivPixelCoordsCols)

            if True:
                ###### For AFMC ########
                for shockLoc in case4['shockReflectionPoints']:
                    plt.axvline(pivPixelCoordsCols[shockLoc],ymin = 0, ymax = 0.5, color = 'r', lw = 0.7, alpha = 0.8)
                for shockLoc in case1['shockReflectionPoints']:
                    plt.axvline(pivPixelCoordsCols[shockLoc],ymin = 0.5, ymax = 1, color = 'r', lw = 0.7, alpha = 0.8)
                # Centre divider
                plt.axhline(pivPixelCoordsRows[case['zeroRow']], color = 'w')
            else:
                for shockLoc in case['shockReflectionPoints']:
                    plt.axvline(pivPixelCoordsCols[shockLoc],ymin = 0.0, ymax = 1, color = 'r', lw = 0.7, alpha = 0.8)

            plt.tight_layout()

            plt.savefig(plotFilename, dpi=dpi)
            plt.close()

    if plotMeanVorticity is True:
        print("Loaded pickle:", pickleFileMeanVorticity)
        [meanW] = pickle.load(open(pickleFileMeanVorticity, "rb"))
        arrays = [meanW]
        filenames = ['Vorticity']
        titles = [r'$\langle \omega \rangle$']
        for array, filename, title in zip(arrays, filenames, titles):
            # Levels (values of the colorbar)
            numLevels = 50
            # Plot the max to the xth std of the field.

            minContourLevel, maxContourLevel = getStdContourLims(array, 4.0)
            levels = np.linspace(minContourLevel, maxContourLevel, numLevels, endpoint=True)
            # Where the labels will actually be placed
            numTicks = 9
            cbarTicks = np.linspace(minContourLevel, maxContourLevel, numTicks, endpoint=True)
            #cbarTicks[1:-1] = np.round(cbarTicks[1:-1], decimals=2)
            # round the points inbetween the min and max
            cbarLabel = title

            plotTitle = 'Vorticity ' + suffixTitle
            plotLocation = picklePlace + 'vorticity/'
            os.makedirs(plotLocation, exist_ok=True)
            plotFilename = plotLocation + 'case' + str(case['case']) + 'vorticity' + filename + suffixFilename + '.png'
            fig = lpt.plotContour(array, plotTitle, levels, cbarLabel, cbarTicks, pivPixelCoordsRows=pivPixelCoordsRows, pivPixelCoordsCols=pivPixelCoordsCols)

            ######################
            # Add a few things for AFMC
            # Plot the shock cell locations
            #[plt.axvline(pivPixelCoordsCols[shockcol], color=(1, 0, 0), alpha=0.3, linewidth = 0.5) for shockcol in case['shockReflectionPoints']]
            for shockLoc in case['shockReflectionPoints']:
                plt.axvline(pivPixelCoordsCols[shockLoc], color='r', lw=0.7, alpha=0.8)
            # Plot the Internozzle lines and entrainment field lines
            #[plt.axhline(pivPixelCoordsRows[standingwaveRow], color=(1, 0, 0), alpha=0.3, linewidth = 0.5) for standingwaveRow in [case['zeroRow'], 40, 660]]
            ######################

            plt.tight_layout()
            plt.savefig(plotFilename, dpi=dpi)
            plt.close()

    if plotStd is True:
        print("Loaded pickle:", pickleFileStd)
        [stdU, stdV, stdFlags] = pickle.load(open(pickleFileStd, "rb"))
        stdArrays = [stdU, stdV, np.sqrt(stdU**2 + stdV**2)]
        stdFilenames = ['u', 'v', 'uv']
        stdTitles = [r'$\log_{10} \left[\sqrt{\langle \left({u^{\prime}/U_e}\right)^2  \rangle} \right]$', r'$\log_{10} \left[\sqrt{\langle \left({v^{\prime}/U_e}\right)^2  \rangle}\right]$', r'$\log_{10} \left[\sqrt{\langle \left({|uv|^{\prime}/U_e}\right)^2  \rangle} \right]$']
        for stdArray, stdFilename, stdTitle in zip(stdArrays, stdFilenames, stdTitles):
            magStd = stdArray
            old = np.seterr(invalid='ignore')
            # Ignore nans for log
            magStd[magStd == 0.0] = np.nan

            magStd = np.log10(np.fliplr(magStd))

            # Levels (values of the colorbar)
            numLevels = 50
            # Plot the max to the xth std of the field.

            minContourLevel, maxContourLevel = getStdContourLims(magStd, 1.4)
            levels = np.linspace(minContourLevel, maxContourLevel, numLevels, endpoint=True)
            # Where the labels will actually be placed
            numTicks = 9
            cbarTicks = np.linspace(minContourLevel, maxContourLevel, numTicks, endpoint=True)
            #cbarTicks[1:-1] = np.round(cbarTicks[1:-1], decimals=2)
            # round the points inbetween the min and max
            cbarLabel = stdTitle

            plotTitle = 'Standard Deviation (' + stdTitle + ' component) ' + suffixTitle
            plotFilename = picklePlace + 'standardDeviation/' + 'case' + str(case['case']) + 'Std' + stdFilename + suffixFilename + '.png'
            fig = lpt.plotContour(magStd, plotTitle, levels, cbarLabel, cbarTicks, pivPixelCoordsRows=pivPixelCoordsRows, pivPixelCoordsCols=pivPixelCoordsCols)

            ######################
            # Add a few things for AFMC
            # Plot the shock cell locations
            #[plt.axvline(pivPixelCoordsCols[shockcol], color=(1, 0, 0), alpha=0.3, linewidth = 0.5) for shockcol in case['shockReflectionPoints']]
            for shockLoc in case['shockReflectionPoints']:
                plt.axvline(pivPixelCoordsCols[shockLoc], color='r', lw=0.7, alpha=0.8)
            # Plot the Internozzle lines and entrainment field lines
            #[plt.axhline(pivPixelCoordsRows[standingwaveRow], color=(1, 0, 0), alpha=0.3, linewidth = 0.5) for standingwaveRow in [case['zeroRow'], 40, 660]]
            ######################

            plt.tight_layout()
            plt.savefig(plotFilename, dpi=dpi)
            plt.close()

    if plotSkewness is True:
        print("Loaded pickle:", pickleFileSkewness)
        [skewU, skewV, skewFlags] = pickle.load(open(pickleFileSkewness, "rb"))
        skewArrays = [skewU, skewV, np.sqrt(skewU**2 + skewV**2)]

        skewFilenames = ['u', 'v', 'uv']
        skewTitles = [r'$u$', r'$v$', r'$|uv|$']
        for skewArray, skewFilename, skewTitle in zip(skewArrays, skewFilenames, skewTitles):
            magSkew = skewArray
            old = np.seterr(invalid='ignore')
            # Ignore nans for log
            magSkew[magSkew == 0.0] = np.nan

            magSkew = np.fliplr(magSkew)
            minContourLevel, maxContourLevel = getStdContourLims(magSkew, 5.0)
            # Levels (values of the colorbar)
            numLevels = 50
            levels = np.linspace(minContourLevel, maxContourLevel, numLevels, endpoint=True)
            # Where the labels will actually be placed
            numTicks = 9
            cbarTicks = np.linspace(minContourLevel, maxContourLevel, numTicks, endpoint=True)

            cbarTicks[1:-1] = np.round(cbarTicks[1:-1], decimals=2)

            # round the points inbetween the min and max
            cbarLabel = r'Skewness' # log _{10}
            plotTitle = 'Skewness (' + skewTitle + ' component) ' + suffixTitle
            plotFilename = picklePlace + 'skewness/' +'case' + str(case['case']) + 'Skewness' + skewFilename + suffixFilename + '.png'
            fig = plotContour(magSkew, plotTitle, levels, cbarLabel, cbarTicks)
            plt.tight_layout()
            plt.savefig(plotFilename, dpi=dpi)
            plt.close()

    if plotKurtosis is True:
        print("Loaded pickle:", pickleFileKurtosis)
        [kurtU, kurtV, kurtFlags] = pickle.load(open(pickleFileKurtosis, "rb"))
        kurtArrays = [kurtU, kurtV, kurtU**2 + kurtV**2]

        kurtFilenames = ['u', 'v', 'uv']
        kurtTitles = [r'$u$', r'$v$', r'$|uv|$']
        for kurtArray, kurtFilename, kurtTitle in zip(kurtArrays, kurtFilenames, kurtTitles):
            magKurt = kurtArray
            old = np.seterr(invalid='ignore')
            # Ignore nans for log
            magKurt[magKurt == 0.0] = np.nan
            magKurt = np.fliplr(magKurt)

            if False: # Display a simple histogram to see the approximate value spread
                hist, bins = np.histogram(magKurt.flatten(), bins=np.arange(50))
                width = 0.7 * (bins[1] - bins[0])
                center = (bins[:-1] + bins[1:]) / 2
                plt.bar(center, hist, align='center', width=width)
                plt.show()

            minContourLevel, maxContourLevel = getStdContourLims(magKurt, 3.0)
            # Levels (values of the colorbar)
            numLevels = 50
            levels = np.linspace(np.nanmin(magKurt), maxContourLevel, numLevels, endpoint=True) #np.nanmax(magKurt)
            # Where the labels will actually be placed
            numTicks = 9
            cbarTicks = np.linspace(np.nanmin(magKurt), maxContourLevel, numTicks, endpoint=True) #np.nanmax(magKurt)

            cbarTicks[1:-1] = np.round(cbarTicks[1:-1], decimals=2)

            # round the points inbetween the min and max
            cbarLabel = r'Kurtosis' #\log _{10} (
            plotTitle = 'Kurtosis (' + kurtTitle + ' component) ' + suffixTitle
            plotFilename = picklePlace + 'kurtosis/' +'case' + str(case['case']) + 'Kurtosis' + kurtFilename + suffixFilename + '.png'
            fig = plotContour(magKurt, plotTitle, levels, cbarLabel, cbarTicks)
            plt.tight_layout()
            plt.savefig(plotFilename, dpi=dpi)
            plt.close()

    if plotRMS is True:
        print("Loaded pickle:", pickleFileRMS)
        [rmsU, rmsV, rmsFlags] = pickle.load(open(pickleFileRMS, "rb"))
        rmsArrays = [rmsU**2, rmsV**2, rmsU**2 + rmsV**2]

        rmsFilenames = ['u', 'v', 'uv']
        rmsTitles = [r'$u$', r'$v$', r'$|uv|$']
        for rmsArray, rmsFilename, rmsTitle in zip(rmsArrays, rmsFilenames, rmsTitles):
            magRMS = np.sqrt(rmsArray)
            old = np.seterr(invalid='ignore')
            # Ignore nans for log
            magRMS[magRMS == 0.0] = np.nan
            magRMS = np.log10(np.fliplr(magRMS))

            #hist, bins = np.histogram(magKurt.flatten(), bins=np.arange(50))
            #width = 0.7 * (bins[1] - bins[0])
            #center = (bins[:-1] + bins[1:]) / 2
            #plt.bar(center, hist, align='center', width=width)
            #plt.show()

            # Levels (values of the colorbar)
            numLevels = 15
            levels = np.linspace(np.nanmin(magRMS), np.nanmax(magRMS), 20, endpoint=True)
            # Where the labels will actually be placed
            numTicks = 9
            cbarTicks = np.linspace(np.nanmin(magRMS), np.nanmax(magRMS), numTicks, endpoint=True)

            cbarTicks[1:-1] = np.round(cbarTicks[1:-1], decimals=2)

            # round the points inbetween the min and max
            cbarLabel = r'$\log _{10} (\mathrm{RMS})$'
            plotTitle = 'RMS (' + rmsTitle + ' component) ' + suffixTitle
            plotFilename = picklePlace + 'rms/' +'case' + str(case['case']) + 'RMS' + rmsFilename + suffixFilename + '.png'
            fig = plotContour(magRMS, plotFilename, plotTitle, levels, cbarLabel, cbarTicks)
            plt.savefig(plotFilename, dpi=dpi)
            plt.close()


    if plot2PointCorr is True:
        print("Loaded pickle:", pickleFile2PtCorr)
        [TwoPtU, TwoPtV, TwoPtFlags, selectPoint] = pickle.load(open(pickleFile2PtCorr, "rb"))
        TwoPtArrays = [TwoPtU, TwoPtV]
        TwoPtFilenames = ['u', 'v']
        TwoPtTitles = [r"$\frac{\langle u_1' u_2' \rangle}{\sqrt{\langle (u'_1)^2\rangle \langle(u'_2)^2\rangle}}$", r"$\frac{\langle v_1' v_2' \rangle}{\sqrt{\langle (v'_1)^2\rangle \langle(v'_2)^2\rangle}}$"]
        for TwoPtArray, TwoPtFilename, TwoPtTitle in zip(TwoPtArrays, TwoPtFilenames, TwoPtTitles):
            magTwoPt = TwoPtArray
            old = np.seterr(invalid='ignore')
            # Ignore nans for log
            magTwoPt[magTwoPt == 0.0] = np.nan

            magTwoPt = np.fliplr(magTwoPt)

            # Levels (values of the colorbar)
            numLevels = 15
            minContourLevel, maxContourLevel = getStdContourLims(magTwoPt, 2.5)
            levels = np.linspace(minContourLevel, maxContourLevel, 30, endpoint=True)
            # Where the labels will actually be placed
            numTicks = 9
            cbarTicks = np.linspace(minContourLevel, maxContourLevel, numTicks, endpoint=True)

            #cbarTicks[1:-1] = np.around(cbarTicks[1:-1], decimals=2)

            # round the points inbetween the min and max


            #plt.xlabel(r'\textbf{time} (s)')
            #plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
            #plt.title(r"\TeX\ is Number "
            #          r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
            #          fontsize=16, color='gray')
            cbarLabel =  TwoPtTitle#r'Pearson\'s correlation coefficient'
            plotTitle = '2 Point Correlation (' + TwoPtTitle + ' component) ' + suffixTitle
            plotFilename = picklePlace + 'twoPointCorr/' +'case' + str(case['case']) + 'TwoPtCorr' + TwoPtFilename + suffixFilename + '.png'
            fig = lpt.plotContour(magTwoPt, plotTitle, levels, cbarLabel, cbarTicks, pivPixelCoordsRows=pivPixelCoordsRows, pivPixelCoordsCols=pivPixelCoordsCols)
            ######################
            # Add a few things for AFMC
            # Plot the shock cell locations
            [plt.axvline(pivPixelCoordsCols[shockcol], color=(1, 0, 0), alpha=0.3, linewidth = 0.5) for shockcol in case['shockReflectionPoints']]

            # Plot the Internozzle lines and entrainment field lines
            #[plt.axhline(pivPixelCoordsRows[standingwaveRow], color=(1, 0, 0), alpha=0.3, linewidth = 0.5) for standingwaveRow in [case['zeroRow'], 40, 660]]
            ######################
            plt.plot(pivPixelCoordsCols[len(pivPixelCoordsCols) - selectPoint[0]],pivPixelCoordsRows[selectPoint[1]], 'ro', markersize = 3.0)
            plt.tight_layout()
            plt.savefig(plotFilename, dpi=dpi)
            plt.close()
    print('Completed', case['case'], datetime.datetime.now())

if __name__ == '__main__':
    picklePlace = './AlkislarStats/'

    # Case and set information is saved in a pickled dicationary
    [case1, case2, case3, case4, case5, case6, case7, case8, case9] = PIVCasesInfo.cases
    cases = [case1, case2, case3, case4, case5, case6, case7, case8, case9]

    #---- Input Options -------#
    cases = [case1, case4]
    cases = [case4]
    # U, V, Flags, vorticity datasets

    axialVel =          True
    transVel =          False
    flags =             False
    vorticity =         False

    # do correlations
    axialAxial =        True
    transTrans =        True
    axialTrans =        False
    vortVort =          False

    daskChunksize = (400, 400, 400)
    if onMassive is True:
        daskChunksize = (1100, 1100, 500)


    doMean =            False
    doStd =             False
    doSkewness =        False
    doKurtosis =        False
    doRMS =             False
    do2PointCorr =      False
    doTemporalPDF =     False
    doMeanBootstrap =   False
    doStdBootstrap =    False
    doMeanBootstrapConfInterval = True
    doStdBootstrapConfInterval = True

    plotSnapshot =      False
    plotVectorSnapshot= False
    plotMean =          False
    plotMeanVorticity = False
    plotStd =           False
    plotSkewness =      False
    plotKurtosis =      False
    plotRMS =           False
    plot2PointCorr =    False

    #--------------------------#

    datasets = np.array(['axialVel', 'transverseVel', 'flags', 'vorticity'])
    datasetsYayOrNay = np.array([axialVel, transVel, flags, vorticity])
    datasets = datasets[datasetsYayOrNay]
    datasetIndexes = np.arange(4)[datasetsYayOrNay]

    # These are used to ref R11, R22, R33 correlations
    corrYayOrNay = np.array([axialAxial, transTrans, axialTrans, vortVort])

    if do2PointCorr is True:
        datasetCorrIndexPairs = np.array([(1, 1), (2, 2), (1, 2), (3, 3)])
        datasetCorrTitlePairs = np.array([[datasets[0]] * 2, [datasets[1]] * 2, [datasets[0], datasets[1]],
                                 [datasets[3]] * 2])

        datasetCorrIndexPairs = datasetCorrIndexPairs[corrYayOrNay]
        datasetCorrTitlePairs = datasetCorrTitlePairs[corrYayOrNay]

    doCalcs = np.array([doMean, doStd, doSkewness, doKurtosis, doRMS, do2PointCorr, doTemporalPDF, doMeanBootstrap,
                        doStdBootstrap, doMeanBootstrapConfInterval, doStdBootstrapConfInterval])
    #doOperations = daskOperations[doCalcs]

    meanQuantity =      'mean'
    stdQuantity =       'std'
    skewnessQuantity =  'skewness'
    kurtosisQuantity =  'kurtosis'
    rmsQuantity =       'rms'

    # Do case-wise statistics
    for case in cases:
        print('#'*60)
        print('case:', case['case'])
        #case['sets'] = case['sets'][:1]
        print('case info', case)
        print('Taking on sets:', case['sets'])
        print('')
        print('Starting case', case['case'], datetime.datetime.now())
        nozzleDiameter = 10 # mm

        suffixFilename = 'Dt' + str(case['dt']).replace('.', '-') + 'NPR' + \
                         str(case['NPR']).replace('.', '-')
        suffixTitle = ' dt ' + str(case['dt']) + ' NPR ' + str(case['NPR'])

        h5CaseDataStoreName = picklePlace + 'case' + str(case['case']) + 'StatStore.h5'
        h5SaveFile = h5py.File(h5CaseDataStoreName, mode='a')

        hdf5OutFileName2PtCorr = picklePlace + 'case' + str(case['case']) + 'TwoPtCorr.h5'

        # PIV data in.
        # On Massive
        if onMassive is True:
            h5BaseFolder = '/home/graham_bell/NCIg75_scratch/gbell/twinJetExp0/'
        # On local Machine
        else:
            h5BaseFolder = '/media/graham/'
        import PIVCasesInfo
        #case1 = PIVCasesInfo.case1['']
        case1 = PIVCasesInfo.case1
        case1Sets = case1['sets']

        h5DataFilePaths = [h5BaseFolder + PIVCasesInfo.setPath(setNumber, absPath=False) +
                           'SUBD/' + 'compiledNC.h5' for setNumber in case['sets']]


        if True:
            # Save a single instantaneous image for the thesis
            daskSaveInstantaneous(dataset='axialVel', h5DataFileNames=h5DataFilePaths, daskChunkSize=daskChunksize)
            exit(0)


        commonOptions = {
            'h5SaveFile': h5SaveFile,
            'datasets': datasets,
            'h5DataFileNames': h5DataFilePaths,
            'daskChunkSize': daskChunksize,
            'suffixFileName': suffixFilename,
            'suffixTitle': suffixTitle,
            'writeGroup':'casewise'
        }
        meanOptions = {
            'operation': daskMean,
            'writeQuantity': meanQuantity
        }
        stdOptions = {
            'operation': daskStd,
            'writeQuantity': stdQuantity
        }
        skewnessOptions = {
            'operation': daskSkewness,
            'writeQuantity': skewnessQuantity
        }
        kurtosisOptions = {
            'operation': daskKurtosis,
            'writeQuantity': kurtosisQuantity
        }
        RMSOptions = {
            'operation': daskRMS,
            'writeQuantity': rmsQuantity
        }

        numResamples = 1000
        meanBootstrapOptions = {
            'operation': daskBootstrap,
            'numResamples': numResamples,
            'datasets': datasets,
            'h5FileNameOut': picklePlace + 'case' + str(case['case']) + 'Bootstrap_Mean.h5',
            'daskfunc': da.nanmean,
            'funcDescript': 'mean',
        }

        stdBootstrapOptions = {
            'operation': daskBootstrap,
            'numResamples': numResamples,
            'datasets':  datasets,
            'h5FileNameOut': picklePlace + 'case' + str(case['case']) + 'Bootstrap_Std.h5',
            'daskfunc': da.nanstd,
            'funcDescript': 'std',
        }

        meanBootstrapConfIntOptions = {
            'operation': calcBootstrapConfInterval,
            # 'datasets':  datasets,
            'sampleStatH5FileLoc': h5CaseDataStoreName,
            'sampleStatLoc': '/casewise/mean/axialVel',
            'bootStrapH5FileLoc': picklePlace + 'case' + str(case['case']) + 'Bootstrap_Mean.h5',
            'bootstrapStdLoc': '/axialVel/bootstrapStd',
            'h5FileNameOut': picklePlace + 'case' + str(case['case']) + 'Bootstrap_Mean.h5',
        }

        stdBootstrapConfIntOptions = {
            'operation': calcBootstrapConfInterval,
            # 'datasets':  datasets,
            'sampleStatH5FileLoc': h5CaseDataStoreName,
            'sampleStatLoc': '/casewise/std/axialVel',
            'bootStrapH5FileLoc': picklePlace + 'case' + str(case['case']) + 'Bootstrap_Std.h5',
            'bootstrapStdLoc': '/axialVel/bootstrapStd',
            'h5FileNameOut': picklePlace + 'case' + str(case['case']) + 'Bootstrap_Std.h5',
        }

        # All the two point correlation stuff is in here.
        if do2PointCorr is True:
            # Use the shear layer data for the points of interest
            shearLayerFileName = 'linkTo_shearLayerData'
    #        shearLayerPoints = pd.read_pickle(shearLayerFileName)

            store = pd.HDFStore(shearLayerFileName + '.h5')
            #store = h5py.File(shearLayerFileName + '.h5', mode='r')
            shearLayerPoints = store['shearLayerPointsDF']
            # Region1 interest in diff between centre and bottom
            # Region2: top
            # Region3 : bottom
            # region 4: top

            #bottom jet E centre
            bottomJetECentreName = 'case' + str(case['case']) + 'Centres' + 'Region' + str(1)
            bottomJetECentres = shearLayerPoints[bottomJetECentreName]

            #bottom jet E outer
            bottomJetEOuterName = 'case' + str(case['case']) + 'bottomEdges' + 'Region' + str(1)
            bottomJetEOuters = shearLayerPoints[bottomJetEOuterName]

            #bottom jet I centre
            bottomJetICentreName = 'case' + str(case['case']) + 'Centres' + 'Region' + str(2)
            bottomJetICentres = shearLayerPoints[bottomJetICentreName]

            # now just do the centres at the same locations.

            # Flip the shear layer points index to reference it from the flipped arrays.
            # Do the shear layer line points.
            # Container to hold multiple lines.
            twoPointRefPointdict = {}
            skipPoint = 5

            twoPointRefPointdict[bottomJetECentreName] = list(zip(np.array(np.round(bottomJetECentres), dtype=int),
                                             1092 - np.array(np.round(shearLayerPoints.index + case['zeroCol']),
                                                             dtype=int)))[::skipPoint]

            twoPointRefPointdict[bottomJetEOuterName] = list(zip(np.array(np.round(bottomJetEOuters), dtype=int),
                                             1092 - np.array(np.round(shearLayerPoints.index + case['zeroCol']),
                                                             dtype=int)))[::skipPoint]

            twoPointRefPointdict[bottomJetEOuterName+'Minus20'] = list(zip(np.array(np.round(bottomJetEOuters)-20, dtype=int),
                                                                 1092 - np.array(
                                                                     np.round(shearLayerPoints.index + case['zeroCol']),
                                                                     dtype=int)))[::skipPoint]

            twoPointRefPointdict[bottomJetICentreName] = list(zip(np.array(np.round(bottomJetICentres), dtype=int),
                                                                1092 - np.array(np.round(shearLayerPoints.index + case['zeroCol']),
                                                                    dtype=int)))[::skipPoint]


            twoPointRefPointdict['symmLine'] = list(zip(np.array([case['zeroRow']]*len(bottomJetICentres),
                                                                 dtype=int),
                                                        1092 - np.array(np.round(shearLayerPoints.index + case['zeroCol']),
                                                                        dtype=int)))[::skipPoint]

            twoPointRefPointdict['EField'] = list(zip(np.array([56] * len(bottomJetICentres), dtype=int),
                                                        1092 - np.array(np.round(shearLayerPoints.index + case['zeroCol']),
                                                                        dtype=int)))[::10]

            TwoPtCorrOptions = {
                'datasets': datasetCorrTitlePairs,
                'datasetIndexes': datasetCorrIndexPairs,
                'h5CorrOutFileName': hdf5OutFileName2PtCorr,
                'refPointDict': twoPointRefPointdict,
                'hdf5FileNames': h5DataFilePaths,
                'chunksize': daskChunksize,
                'suffixFileName': suffixFilename,
                'suffixTitle': suffixTitle,
                'operation': spatialCorrManager,
            }


            temporalPDFOptions = {
                'datasets': datasets,
                'datasetIndexes': datasetIndexes,
                'hdf5FileNames': h5DataFilePaths,
                'chunksize': daskChunksize,
                'suffixFileName': suffixFilename,
                'suffixTitle': suffixTitle,
                'operation': daskTemporalPDF,
                'refPoints': [[229, 1000], [229, 960], [229, 850], [229, 750], [229, 650], [229, 550], [229, 420]]
            }
        else:
            TwoPtCorrOptions = {}
            temporalPDFOptions = {}

        dicts = np.array([meanOptions, stdOptions, skewnessOptions, kurtosisOptions,
                          RMSOptions, TwoPtCorrOptions, temporalPDFOptions, meanBootstrapOptions, stdBootstrapOptions,
                          meanBootstrapConfIntOptions, stdBootstrapConfIntOptions])
        # Confusing: Grab dicts from the binary doCalcs array.
        dicts = dicts[doCalcs]

        for userDict in dicts:
            #Not python 2 friendly on massive
            #compiledDict = {**commonOptions, **userDict}
            compiledDict = commonOptions.copy()
            compiledDict.update(userDict)
            runOperationAndSave(compiledDict)

        h5SaveFile.close()
        print('Completed case', case['case'], 'at', datetime.datetime.now())
        print('')
        print('#'*60)


