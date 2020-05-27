# Implimentation of POD in PIV
#
import PIVCasesInfo
from PIVCasesInfo import cases
from PODClass import PODComputer
import sys
import numpy as np
import os
import time
import dask.array as da
import h5py
import sys
sys.path.append("../PlotTools/")
import latexPlotTools as lpt

def buildArraySlices(case):
    fullSlice = {
        'colMin': None,
        'colMax': None,
        'rowMin': None,
        'rowMax': None,
        'suffix': '_fullfield'
    }

    zeroColSlice = {
        'colMin': case['zeroCol'],
        'colMax': None,
        'rowMin': None,
        'rowMax': None,
        'suffix': '_zeroCol'
    }

    postMachDiskSlice = {
        'colMin': case['zeroCol'] + 100,
        'colMax': None,
        'rowMin': None,
        'rowMax': None,
        'suffix': '_fullSlice_pastMachDisk'
    }

    lowerJetSlice = {
        'colMin': case['zeroCol'],
        'colMax': None,
        'rowMin': None,
        'rowMax': case['zeroRow'],
        'suffix': '_lowerJet'
    }

    upperJetSlice = {
        'colMin': case['zeroCol'],
        'colMax': None,
        'rowMin': case['zeroRow'],
        'rowMax': None,
        'suffix': '_upperJet'
    }

    lowerJetSlicePastMachDisk = {
        'colMin': case['zeroCol'] + 100,
        'colMax': None,
        'rowMin': None,
        'rowMax': case['zeroRow'],
        'suffix': '_lowerJetPastMachDisk'
    }

    upperJetSlicePastMachDisk = {
        'colMin': case['zeroCol'] + 100,
        'colMax': None,
        'rowMin': case['zeroRow'],
        'rowMax': None,
        'suffix': '_upperJetPastMachDisk'
    }

    sliceDict = {
        'fullSlice': fullSlice,
        'zeroColSlice': zeroColSlice,
        'postMachDiskSlice': postMachDiskSlice,
        'lowerJetSlice': lowerJetSlice,
        'upperJetSlice': upperJetSlice,
        'lowerJetSlicePastMachDisk': lowerJetSlicePastMachDisk,
        'upperJetSlicePastMachDisk': upperJetSlicePastMachDisk
    }
    return sliceDict

if __name__ == '__main__':


    # Perform simple example first on 1 set to get idea
    [case1, case2, case3, case4, case5, case6, case7, case8, case9] = cases

    #--------------Input---------------------
    # Perform POD on which field?
    uField =                    True
    vField =                    True
    doPOD =                     True
    # Turn off when running on Massive
    plotEnergyDist =            True
    plotModes =                 False
    plotSnapshotJointScatter =  False
    plotintermittencyContour =  False

    onMassive = False
    # If on Massive
    if onMassive is True: #sys.version_info[0] == 2:
        # can't plot via massive, turn off plotting.
        plotEnergyDist =            True
        plotModes =                 False
        plotSnapshotJointScatter =  False
        plotintermittencyContour =  False
        h5BaseFolder = '/home/grahamb/le11_scratch/gbell/twinJetExp0/'
        PODBaseFolder = './PODModes_Massive/'
        # Do which data sets?
        setRange = (None, None)
        numModes = 20

    else:
        h5BaseFolder = '/media/graham/'
        # Do which data sets?
        setRange = (2, 3)
        # PODBaseFolder = './PODModes_inMem/'
        PODBaseFolder = './PODModes_Massive/'

        numModes = 20

    #----------------------------------------

    fields = np.array([uField, vField], dtype=np.bool)
    datasets = np.array(['axialVel', 'transverseVel'])
    datasets = datasets[fields]
    fieldNames = np.array(['U', 'V'])
    fieldNames = fieldNames[fields]
    fileFieldName = ''.join(fieldNames)

    # Mode pairs on pastMachDisk slice.
    case1['modePairs'] = [(0, 1), (2, 3)]
    case4['modePairs'] = [(0, 1), (2, 3), (4, 5), (6, 7)]
    # cases = [case1, case4]
    cases = [case1]
    # cases = [case4]


    for case in cases:
        print()
        print('#' * 60)
        print('Starting case:', case['case'])

        sliceDict = buildArraySlices(case=case)
        # slices = [fullSlice, zeroColSlice, postMachDiskSlice, lowerJetSlice, lowerJetSlicePastMachDisk]
        # slices = [lowerJetSlice]
        # slices = [fullSlice]
        # slices = [upperJetSlice]
        # slices = [postMachDiskSlice]
        # slices = [zeroColSlice]
        # slices = [fullSlice, lowerJetSlicePastMachDisk, upperJetSlicePastMachDisk]
        slices = [sliceDict['fullSlice'], sliceDict['lowerJetSlicePastMachDisk'], sliceDict['upperJetSlicePastMachDisk']]
        # slices = [sliceDict['lowerJetSlicePastMachDisk'], sliceDict['upperJetSlicePastMachDisk']]
        # slices = [sliceDict['fullSlice'], ]

        # for slice in slices:
        sliceNames = ['fullSlice', 'lowerJetSlicePastMachDisk', 'upperJetSlicePastMachDisk']
        # for slice in slices:
        for sliceName in sliceNames:
            slice = sliceDict[sliceName]
            print('Starting Slice:', slice['suffix'])

            ###### Slicing ######
            # If you use -1 here you will exclude the last row/element. Use None to start/end at boundaries.
            colMin = slice['colMin']
            colMax = slice['colMax']
            rowMin = slice['rowMin']
            rowMax = slice['rowMax']
            #####################

            ##########################################################################
            print('Working in folder:')

            resultsFolder = PODBaseFolder + 'case' + str(case['case']) + fileFieldName
            print(resultsFolder)

            if 'suffix' in slice:
                resultsFolder += slice['suffix']

            resultsFolder += '/'

            if sys.version_info[0] == 2:
                # Check if dir exists and make (version for python2)
                print('Taking python2 route')
                if not os.path.exists(resultsFolder):
                    os.makedirs(resultsFolder)
            else:
                print('Taking python3 route')
                os.makedirs(resultsFolder, exist_ok=True)

            suffixFilename = 'Dt' + str(case['dt']).replace('.', '-') + \
                             'NPR' + str(case['NPR']).replace('.', '-')
            suffixTitle = ' dt ' + str(case['dt']) + ' NPR ' + str(case['NPR'])
            pickledModeCoeffFileName = resultsFolder + suffixFilename + fileFieldName + \
                                       'PODModeCoefficients' + '.dat'
            ##########################################################################

            POD = PODComputer(resultsFolder=resultsFolder,
                              case=case,
                              fileSuffix=suffixFilename,
                              datasets=datasets,
                              fields=fields,
                              fieldNames=fieldNames)

            if doPOD is True:
                # Build the data from h5filenames.
                print('########################## Timer start ############################')
                startTimer = time.clock()
                print('Starting read...')

                # 3 Random sets
                useSets = case['sets'][setRange[0]:setRange[1]]
                hdf5Files = [h5BaseFolder + PIVCasesInfo.setPath(setNumber, absPath=False) + 'SUBD/' + 'compiledNC.h5'
                             for setNumber in useSets]

                # print(hdf5Files)
                # Pack and form a single array with Dask
                chunksize = (400, 400, 500)
                data = []
                for dataset in datasets:
                    datasetData = [da.from_array(h5py.File(file, 'r')[dataset], chunks=chunksize) for file in hdf5Files]
                    datasetData = da.concatenate(datasetData, axis=2)
                    print('Dataset size:', datasetData.shape)
                    print('Reading', dataset)
                    readData = np.array(datasetData, dtype=np.float32)
                    # Flip the data horizontally
                    readData = readData[:, ::-1, :]
                    print('Done.')

                    if False:
                        # Old way, manually actually chopping the data.
                        # Slice the data #
                        readData = readData[rowMin:rowMax, colMin:colMax, :]

                    if True:
                        mask = np.full(shape=readData.shape, fill_value=False)
                        mask[rowMin:rowMax, colMin:colMax, :] = True
                        readData[mask == False] = 0.0

                    # Get the shape of the original data.
                    vectorShape = readData.shape

                    data.append(readData)
                print('Done read.')
                # lpt.vectorDirectionalColor(xArray=data[0][:, :, 0], yArray=data[1][:, :, 0])
                # exit(0)

                # Uall = da.vstack(data)
                Uall = np.vstack(data)
                del data, readData, datasetData, mask

                #Down sampling
                if False:
                    Uall = np.nan_to_num(Uall)
                    from scipy.ndimage.filters import gaussian_filter
                    for i in range(Uall.shape[2]):
                        if i % 50 == 0:
                            print('field', i, 'of', Uall.shape[2], 'blurred.')
                        field = Uall[:, :, i]
                        blurred = gaussian_filter(field, sigma=7)
                        # blurred = blurred[::2, ::2]
                        Uall[:, :, i] = blurred

                if True:
                    #some how still a few nans, reflash these.
                    # Convert nans to 0.0
                    Uall = np.nan_to_num(Uall)

                # Compute mean in dask to save doubling data in memory
                print('Starting dask mean operation')
                Uall_dask = da.from_array(Uall, chunks=(1e3, 1e3, 1e3))
                UallMean = da.nanmean(Uall_dask, axis=2).compute()
                print('Done.')

                Uall = Uall - UallMean[:, :, np.newaxis] #np.nanmean(Uall, axis=2)[:, :, np.newaxis]

                if False:
                    # A look at the temporal direction
                    lpt.vectorDirectionalColor
                    plt.pcolormesh(Uall[:, :, 3])
                    plt.colorbar()
                    plt.plot(450, 135, 'r.')
                    plt.show()

                    exit(0)

                if False:
                    # Save out a quick dataset for Joel.
                    h5FName = 'case' + str(case['case']) + fileFieldName + '.h5'
                    h5SaveFile = h5py.File(h5FName, mode='w')
                    dset = h5SaveFile.create_dataset(name='flucs', data=Uall, shape=Uall.shape, dtype='float32')
                    h5SaveFile.flush()
                    h5SaveFile.close()
                    exit(0)

                POD.computePOD(Uall=Uall, saveData=True, vectorShape=vectorShape)
                del Uall
                print('################################ End timer ###############################')
                endTimer = time.clock()
                print('Run time:', endTimer-startTimer)

            ##############################
            #Load the saved data
            eigenValues = POD.loadEigValues()

            if plotEnergyDist is True:
                # For the lower and upper energy plots, plot them side by side to save space.
                # Paper plotting
                if False:
                    if sliceName == sliceNames[1] or sliceName == sliceNames[2]:
                        width = 0.49
                        square = True
                    else: # For the fullfield it's ok to plot it as a 0.8
                        width = 0.8
                        square = False
                if True: # All rectangular for JFM resposne.
                    width = 0.8
                    square = False

                figSize = lpt.figsize(width=width, fig_width_pt=384.0)
                # Adjust the height to do the squashed version for subplotting.# Make a square plot
                if square is True:
                    figSize[1] = figSize[0]
                else:
                    figSize[1] = figSize[0] * 0.4
                POD.plotEnergyDist(figSize=figSize, legend=False)

            if plotModes is True:
                modeCoords = case['pivPixelCoordsRows'], case['pivPixelCoordsCols']
                # Plotted side by side

                if len(datasets)> 1:
                    figWidth = 1.0
                    subplotAdjustInput = {
                        'right': 0.8,
                        'bottom': 0.22,
                        'left': 0.1,
                        'top': 0.99,
                        'wspace':0.3,
                    }
                    figSize = lpt.figsize(width=figWidth, fig_width_pt=384.0)
                    figSize[1] = figSize[0] * 0.3

                else:
                    figWidth = 0.49
                    subplotAdjustInput = {
                        'right': 0.8,
                        'bottom': 0.20,
                        'left': 0.21,
                        'top': 0.95
                    }
                    figSize = lpt.figsize(width=figWidth, fig_width_pt=384.0)

                POD.plotModes(modeCoords=modeCoords, figSize=figSize, title=False,
                              subplotAdjustInput=subplotAdjustInput)

            if plotSnapshotJointScatter is True:
                POD.plotSnapshotJointScatter()

            if plotintermittencyContour is True:
                POD.plotintermittencyContour()

            del POD, eigenValues