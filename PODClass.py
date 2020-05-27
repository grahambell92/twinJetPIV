import h5py
import numpy as np
import dask.array as da


import compileNC
import PIVCasesInfo
# from PIVCasesInfo import cases
import ideallyExpandedTools
import sys
import scipy
import itertools
sys.path.append("../PlotTools/")
# Custom plotting tools
import latexPlotTools as lpt
from scipy import interpolate
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import itertools as it


class PODComputer(object):
    def __init__(self, resultsFolder, case, numModes=20, fileSuffix=None, datasets=None, fields=None, fieldNames=None):
        self.resultsFolder = resultsFolder
        self.numModes = numModes
        self.case = case
        self.fileSuffix = fileSuffix

        self.datasets = datasets
        self.fields = fields
        self.fieldNames = fieldNames
        self.h5Store = h5py.File(resultsFolder+'PODresults.h5', mode='a')

        self.PODModeCoeff_h5name = 'PODModeCoeff'
        self.PODEnergy_h5name = 'PODEnergy'
        self.PODEnergy_h5name = 'PODEnergy'
        self.PODModes_h5Folder = 'PODModes/'

    def normaliseArray(self, array, newMin=-1.0, newMax=1.0):
        newArray = (newMax - newMin) / (np.nanmax(array) - np.nanmin(array)) * (array - np.nanmax(array)) + newMax
        # 123 = (max'-min')/(max-min)*(value-max)+max'
        return newArray

    def interpHolesWorker(self, slice):
        # Example taken from https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
        # index = indexAndSliceTuple[0]
        # slice = indexAndSliceTuple[1]
        print('taking on slice...')
        x = np.arange(0, slice.shape[1])
        y = np.arange(0, slice.shape[0])
        # mask invalid values
        currentField = np.ma.masked_invalid(slice)
        xx, yy = np.meshgrid(x, y)
        # get only the valid values
        x1 = xx[~currentField.mask]
        y1 = yy[~currentField.mask]
        newarr = currentField[~currentField.mask]
        GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
        return GD1

    def interpHolesManager(self, vectorArray):
        print('Starting parallel interpolation')
        # iterating over a 3d will go through the third axis
        import multiprocessing
        numProcess = multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(numProcess)
        pool = multiprocessing.Pool(2)

        returnedArray = pool.map(func=self.interpHolesWorker, iterable=vectorArray[:, :, :6].transpose(2, 0, 1))
        returnedArray = np.dstack(returnedArray)
        for slice in returnedArray.transpose(2, 0, 1):
            plt.contourf(slice)
            plt.show()
        exit(0)

    def interpolateHoles(self, vectorArray):
        print('Starting interpolation...')
        for i in range(vectorArray.shape[2]):
            print('field', i, 'of', vectorArray.shape[2])
            # Convolve filter with ones will produce value where nans are isolated on all edges.
            currentField = vectorArray[:, :, i]
            # nanLocations = np.isnan(currentField)

            if False:
                xx, yy = np.meshgrid(x, y)
                z = np.sin(xx ** 2 + yy ** 2)
                f = interpolate.interp2d(x, y, z, kind='cubic')

                xnew = np.arange(-5.01, 5.01, 1e-2)
                ynew = np.arange(-5.01, 5.01, 1e-2)
                znew = f(xnew, ynew)

            if True:
                # Example taken from https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
                x = np.arange(0, currentField.shape[1])
                y = np.arange(0, currentField.shape[0])
                # mask invalid values
                currentField = np.ma.masked_invalid(currentField)
                xx, yy = np.meshgrid(x, y)
                # get only the valid values
                x1 = xx[~currentField.mask]
                y1 = yy[~currentField.mask]
                newarr = currentField[~currentField.mask]
                GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')

            vectorArray[:, :, i] = GD1
        return vectorArray

    def computePOD(self, Uall, vectorShape, saveData=False):

        (UallshapeRows, UallshapeCols, UallshapeTime) = Uall.shape
        Uall = np.reshape(Uall, (UallshapeRows * UallshapeCols, UallshapeTime))

        print('Starting auto-correlation matrix')
        # acm = auto-correlation/covariance matrix
        autoCorrelationMatrix = np.dot(Uall.T, Uall)
        # autoCorrelationMatrix = da.dot(da.transpose(Uall_dask), Uall).compute()
        print('Done.')

        print('Starting Eigen value problem...')
        eigenValues, eigenVectors = scipy.linalg.eig(autoCorrelationMatrix)
        print('Done.')

        del autoCorrelationMatrix

        # Sort largest to smallest
        # Get the reverse of the indexes
        # Flash the eigenValues to a magnitude from complex.
        eigenValues = np.abs(eigenValues)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        # Pickle out the mode energy (eigen values).

        # Compute modes
        if True:
            PODmodes_vec = np.empty((UallshapeRows * UallshapeCols, self.numModes))
            PODmodes = np.empty((UallshapeRows, UallshapeCols, self.numModes))

            print('Starting mode shape calculation...')
            for i in range(self.numModes):
                # if i % 10 == 0:
                #     print('mode', i, 'of', self.numModes)
                # +1 to the eigenVectors because first mode is mean vel
                print('mode', i, 'of', self.numModes)

                tmp = np.dot(Uall, eigenVectors[:, i])
                mode = tmp / np.linalg.norm(tmp)
                mode = mode.real
                PODmodes_vec[:, i] = mode

                mode = mode.reshape((UallshapeRows, UallshapeCols))
                PODmodes[:, :, i] = mode

        if saveData is True:
            if self.PODEnergy_h5name in self.h5Store:
                del self.h5Store[self.PODEnergy_h5name]
            self.h5Store[self.PODEnergy_h5name] = eigenValues
            self.h5Store.flush()

            if True:
                # Calculate and Pickle out modes/eigen vectors.
                # modeGenerator = computeModes()
                for i in range(self.numModes):
                    mode = PODmodes[:, :, i]
                    publishedModeNum = i + 1

                    if len(self.datasets) > 1:
                        (vectorShapeRows, vectorShapeCols, vectorShapeTime) = vectorShape
                        modeU = mode[:vectorShapeRows, :]
                        modeV = mode[vectorShapeRows:, :]
                        modeU_h5Name = self.PODModes_h5Folder + self.fieldNames[0] + '/' + str(publishedModeNum)
                        modeV_h5Name = self.PODModes_h5Folder + self.fieldNames[1] + '/' + str(publishedModeNum)

                        if modeU_h5Name in self.h5Store:
                            del self.h5Store[modeU_h5Name]
                        self.h5Store[modeU_h5Name] = modeU

                        if modeV_h5Name in self.h5Store:
                            del self.h5Store[modeV_h5Name]
                        self.h5Store[modeV_h5Name] = modeV

                    else:
                        modeX_h5Name = self.PODModes_h5Folder + self.fieldNames[0] + '/' + str(publishedModeNum)
                        if modeX_h5Name in self.h5Store:
                            del self.h5Store[modeX_h5Name]
                        self.h5Store[modeX_h5Name] = mode

                    self.h5Store.flush()
                del eigenVectors#, eigenValues


                # Now loop over all the snapshots for modes 0 and 1 to produce the coeff
                if True:
                    print('Starting mode coefficient calculation...')
                    PODModes_dask = da.from_array(PODmodes_vec, chunks=(1e5, 1e5))
                    Uall_dask = da.from_array(Uall, chunks=(1e5, 1e5))
                    modeCoeffStore = da.dot(PODModes_dask.T, Uall_dask).compute()
                    print('CoeffStore', modeCoeffStore.shape)
                    if self.PODModeCoeff_h5name in self.h5Store:
                        del self.h5Store[self.PODModeCoeff_h5name]
                    self.h5Store[self.PODModeCoeff_h5name] = modeCoeffStore
                    self.h5Store.flush()
                    print('Done.')

        return PODmodes, eigenValues

    def loadEigValues(self):
        ##############################
        # Load the pickled Data
        # Massive has only python2. Pickle files are different.
        eigenValues = np.array(self.h5Store[self.PODEnergy_h5name])
        print('Done.')
        print()
        # set last eigenvalue to 0
        eigenValues[-1] = 0.0
        # Relative energy
        self.modeEnergy = eigenValues / np.sum(eigenValues)
        ##############################
        return self.modeEnergy

    def loadEigVecs(self, fieldName, returnArray=False):
        dataFolder = self.PODModes_h5Folder + fieldName + '/'
        modeHandles = [self.h5Store[dataFolder+'/' + str(i)] for i in range(self.numModes)
                       if str(i) in self.h5Store[dataFolder]]
        if returnArray is True:
            modes = [modeHandle[:] for modeHandle in modeHandles]
            # modes = np.dstack(modes)
            return modes
        return modeHandles

    def loadModeCoeffs(self, norm=False):
        modeCoeffs = self.h5Store[self.PODModeCoeff_h5name][:]
        if norm is True:
            modeCoeffs = self.normaliseArray(array=modeCoeffs)
        return modeCoeffs

    def plotEnergyDist(self, figSize=None, subplotAdjustInput=None, legend=True):

        if figSize is not None:
            fig = plt.figure(1, figsize=figSize)
        else:
            fig = lpt.getStandardFigureSize()

        ax = fig.add_subplot(1, 1, 1)
        modeNums = np.arange(1, len(self.modeEnergy) + 1)
        ax.semilogx(modeNums, self.modeEnergy * 100, color='0.0', marker='o', ls='None', markersize=3,
                    markeredgewidth=0.0, label=r'Sp. KE \%')
        # ax.set_ylim(ymin=1e-3)
        ax.set_xlabel(r'Mode number')
        ax.set_ylabel(r'Sp. KE \%')
        ax2 = ax.twinx()

        ax2.semilogx(modeNums, np.cumsum(self.modeEnergy) * 100, ls='-', color='0.0', linewidth=1.0,
                     label=r'Cum. sp. KE \%')
        ax2.set_ylabel('Cum. sp. KE \%')
        ax.set_xlim(xmin=0.9)

        if legend is True:
            handles_ax, labels_ax = ax.get_legend_handles_labels()
            handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
            labels = np.array(labels_ax + labels_ax2)
            handles = np.array(handles_ax + handles_ax2)
            lgnd = plt.legend(handles, labels, loc=9, frameon=False, )  # bbox_to_anchor=(0.4, -0.35))

        # plt.title(suffixTitle + ' POD Mode energy contribution. ' + str(UallshapeTime) + ' snapshots.')
        if subplotAdjustInput is not None:
            fig.subplots_adjust(**subplotAdjustInput)
        else:
            plt.tight_layout()
        plotFileName = self.resultsFolder + 'PODEnergy' + '.png'
        plt.savefig(plotFileName, dpi=300)
        #                    bbox_extra_artists=(lgnd,), bbox_inches='tight')
        plt.close('all')
        print('Done.')

        return

    def plotModes(self, modeCoords=None, figSize=None, title=True, subplotAdjustInput=None):

        dataset0Folder = self.PODModes_h5Folder + self.fieldNames[0] + '/'
        for i in range(1, self.numModes):
            if str(i) in self.h5Store[dataset0Folder]:
                modeH5Handles = [self.h5Store[self.PODModes_h5Folder + fieldName + '/' + str(i)] for fieldName in self.fieldNames]

                if figSize is not None:
                    fig = plt.figure(1, figsize=figSize)
                else:
                    fig = lpt.getStandardFigureSize(numOfCols=len(self.datasets))

                gridLayout = gridspec.GridSpec(1, len(modeH5Handles))
                setLeft_yAxisLabel = True
                for plotCol, (modeHandle, fieldName) in enumerate(zip(modeH5Handles, self.fieldNames)):
                    mode = modeHandle[:]
                    ax = plt.subplot(gridLayout[0, plotCol])
                    # modeTitle = np.array([r'$u_x$', r'$u_y$'])[plotCol]
                    modeTitle = fieldName

                    # mode = mode.real
                    # Normalise mode to -1 to 1
                    mode = self.normaliseArray(mode, newMin=-1.0, newMax=1.0)
                    # mode = normaliseArray(array=mode, newMin=-1.0, newMax=1.0)

                    minContourLevel = np.nanmin(mode)
                    maxContourLevel = np.nanmax(mode)
                    if modeCoords is not None:
                        pivPixelCoordsRows, pivPixelCoordsCols = modeCoords[0], modeCoords[1]
                    else:
                        pivPixelCoordsRows = np.arange(mode.shape[0])
                        pivPixelCoordsCols = np.arange(mode.shape[1])

                    # levels = np.linspace(minContourLevel, maxContourLevel, 30)

                    cont = ax.pcolormesh(pivPixelCoordsCols, pivPixelCoordsRows, mode, cmap='inferno')
                    ax.set_aspect('equal')
                    if title is True:
                        ax.set_title(modeTitle)
                    ax.set_xlabel(r'$x/D$')
                    if setLeft_yAxisLabel is True:
                        ax.set_ylabel(r'$y/D$')
                        setLeft_yAxisLabel = False

            if title is True:
                titleText = r''.join(['Mode ', str(i), ', mode energy: ',
                                      '{:.2f}'.format(self.modeEnergy[i-1] * 100.0), r'~\%'])
                fig.suptitle(titleText)
            #
            # fig.subplots_adjust(left=0.1,
            #                     right=0.4,
            #                     top=0.99,
            #                     bottom=0.1)
            left=0.85
            bottom = 0.26
            width=0.05
            height=0.68
            cbar_ax = fig.add_axes([left, bottom, width, height])
            cbar = fig.colorbar(cont, cax=cbar_ax)

            # cbar = plt.colorbar(cont, ax=ax)
            numTicks = 3
            cbarTicks = np.linspace(minContourLevel, maxContourLevel, numTicks, endpoint=True)
            cbar.set_ticks(cbarTicks)
            cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in cbarTicks])
            cbar.set_label(r'$\phi$')
            # plt.tight_layout()
            if subplotAdjustInput is not None:
                fig.subplots_adjust(**subplotAdjustInput)

            plotName = 'PODMode' + str(i) + '.png'
            if self.fileSuffix is not None:
                plotName = self.fileSuffix + plotName
            plotName = self.resultsFolder + plotName
            plt.savefig(plotName, dpi=300)
            print('Plotting Mode:', plotName)
            plt.close('all')
            print('Plotted mode:', i)



    def plotSnapshotJointScatter(self):
        modeCoeffs = self.h5Store[self.PODModeCoeff_h5name]
        modeCoeffs = self.normaliseArray(array=modeCoeffs)

        # mode0Coeffs = modeCoeffs[0, :]
        for modeX, modeY in self.case['modePairs']:
            modeXCoeffs = modeCoeffs[modeX, :]
            modeYCoeffs = modeCoeffs[modeY, :]

            fig = lpt.getStandardFigureSize()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(modeXCoeffs, modeYCoeffs, marker='.', linestyle='None',
                    markersize=2, markeredgecolor='None', alpha=0.8)
            ax.set_xlabel('Mode ' + str(modeX+1))
            ax.set_ylabel('Mode ' + str(modeY+1))
            ax.set_aspect('equal')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            plt.tight_layout()
            # ax.set_xlim(-1000, 1000)
            # ax.set_ylim(-1000, 1000)

            plotName = 'modeScatter_fields' + ''.join(self.fieldNames) + \
                       '_pair' + str(modeX+1) + '-' + str(modeY+1) + '.png'

            if self.fileSuffix is not None:
                plotName = self.fileSuffix + plotName
            plotName = self.resultsFolder + plotName

            print(plotName)
            plt.savefig(plotName, dpi=300)
            plt.close('all')

    def plotintermittencyContour(self):
        modeCoeffs = self.h5Store[self.PODModeCoeff_h5name]
        modeCoeffs = self.normaliseArray(array=modeCoeffs)

        # Do all the combinations of the lists together.
        for (modeA0, modeA1), (modeB0, modeB1) in it.combinations(self.case['modePairs'], r=2):
            modeA0Coeffs = modeCoeffs[modeA0, :]
            modeA1Coeffs = modeCoeffs[modeA1, :]
            modeB0Coeffs = modeCoeffs[modeB0, :]
            modeB1Coeffs = modeCoeffs[modeB1, :]

            # AvgModeAEnergy = 100*(mEnergy[modeA0] + mEnergy[modeA1]) / 2.0
            # AvgModeBEnergy = 100*(mEnergy[modeB0] + mEnergy[modeB1]) / 2.0

            modeAmag = np.sqrt(modeA0Coeffs ** 2 + modeA1Coeffs**2)   # /AvgModeAEnergy
            modeBmag = np.sqrt(modeB0Coeffs ** 2 + modeB1Coeffs**2)     #/AvgModeBEnergy

            magA3rdQuartile = np.percentile(modeAmag, q=75)
            magB3rdQuartile = np.percentile(modeBmag, q=75)

            greaterThanA = np.greater(modeAmag, magA3rdQuartile)
            lessThanA = np.less(modeAmag, magA3rdQuartile)
            greaterThanB = np.greater(modeBmag, magB3rdQuartile)
            lessThanB = np.less(modeBmag, magB3rdQuartile)

            greaterThanAB = np.logical_and(greaterThanA, greaterThanB)
            greaterThanA_lessThanB = np.logical_and(greaterThanA, lessThanB)
            greaterThanB_lessThanA = np.logical_and(greaterThanB, lessThanA)

            print(np.sum(greaterThanAB))
            print(np.sum(greaterThanA_lessThanB))
            print(np.sum(greaterThanB_lessThanA))
            print(np.sum(greaterThanAB) / np.sum(greaterThanB_lessThanA))

            fig = lpt.getStandardFigureSize()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(modeAmag, modeBmag, marker='.', linestyle='None', markersize=2, markeredgecolor='None',
                    alpha=0.4, )
            ax.axvline(magA3rdQuartile, linestyle='--', color='0.5', alpha=0.5)
            ax.annotate('Q3', xy=(magA3rdQuartile, 0.9), color='0.5', alpha=0.5)
            ax.axhline(magB3rdQuartile, linestyle='--', color='0.5', alpha=0.5)
            ax.annotate('Q3', xy=(0.9, magB3rdQuartile), color='0.5', alpha=0.5)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(r'$|$ Mode pair: ' + str(modeA0+1) + ' \& ' + str(modeA1+1) + '$|$')
            ax.set_ylabel(r'$|$ Mode pair: ' + str(modeB0+1) + ' \& ' + str(modeB1+1) + '$|$')
            ax.set_aspect('equal')
            plotName = self.resultsFolder
            if self.fileSuffix is not None:
                plotName += self.fileSuffix + '_'
            plotName += 'fields' + ''.join(self.fieldNames) + \
                       '_modePairConcurrency' + '_pair' + str(modeA0+1) + '-' + str(modeA1+1) + '_vs_' + str(modeB0+1) + \
                       '-' + str(modeB1+1) + '.png'
            plt.tight_layout()
            plt.savefig(plotName, dpi=300)
            plt.close('all')