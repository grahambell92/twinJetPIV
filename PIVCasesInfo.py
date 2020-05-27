# File that has a dictionary of all the cases from the Experimental Twin Jet campaign 0.
# Written by Graham Bell 18/05/2016.

# ----------------------------------
## 2020 archival update by Graham Bell
#

# ----------------------------------

import pickle
import sys
import compileNC
import numpy as np

def setPath(setNumber, absPath=True):
    GrahamDrive0 = 'GrahamDrive0/Far2016/'
    GrahamDrive1 = 'GrahamDrive1/Far2016/'
    GrahamDrive2 = 'GrahamDrive2/Far2016/'
    GrahamDrive3 = 'GrahamDrive3/Far2016/'
    if setNumber >= 1 and setNumber <= 25:
        path = GrahamDrive0 + 'Set' + str(setNumber) + '/'
    elif setNumber >= 26 and setNumber <= 57:
        path = GrahamDrive1 + 'Set' + str(setNumber) + '/'
    elif setNumber >= 58 and setNumber <= 75:
        path = GrahamDrive2 + 'Set' + str(setNumber) + '/'
    elif setNumber >= 76 and setNumber <= 98:
        path = GrahamDrive3 + 'Set' + str(setNumber) + '/'
    else:
        print("No path for Set number known.")
        return
    if absPath is True:
        # If on local PhD machine
        path = '/media/graham/' + path
        # if on Massive
        # path = '/home/grahamb/le11_scratch/gbell/twinJetExp0/' + path

    return path

def getRealWorldCoords(case, nozzleDiameter=10):
    #nozzleDiameter = 10  # mm
    sampleNC = setPath(case['sets'][0]) + 'SUBD/' + 'subd00000008.nc'
    pivPixelCoordsRows, pivPixelCoordsCols = compileNC.getNCCoords(sampleNC)
    pivPixelCoordsRows = case['mag'] * 1000 * np.asarray(
        pivPixelCoordsRows - pivPixelCoordsRows[case['zeroRow']]) / nozzleDiameter
    pivPixelCoordsCols = case['mag'] * 1000 * np.asarray(
        pivPixelCoordsCols - pivPixelCoordsCols[case['zeroCol']]) / nozzleDiameter
    return pivPixelCoordsRows, pivPixelCoordsCols

# Case 1, dt = 0.9us, press = 1.866V, Type: Farfield, Position : Centre:
centreMag = 19.9e-6  # m/px
leftMag = 19.1e-6  # m/px
case1 = {'case': 1,
         'dt': 0.9,  # us
         'pressure': 1.866,
         'NPR': 4.60,
         'mag': centreMag,
         'calcNozzleExitVel': 310.5,  # m/s
         'type': 'F', # F is for farfield meaning that the camera can see the jets and the far field around them.
         'pos': 'C', # C is for looking at the centre of the symmetry plane between the jes.
         'zeroRow': 360, # This is the row of the y=0 position.
         'zeroCol': 58, # This is the row of the x=0 position.
         'shockReflectionPoints': [195, 329, 465, 600, 725, 835], # These are the indices for the axial jet boundary shock reflection points.
         'sets': [2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29], # Bad sets in statistics, 1, 77, 78 # These are the sets included in this case.
         'badImgCutOff': 0.08, # This is used in the prePIV image processing routine.
         # Set 18 has avg 7% interpolation. Removed 18, #
         'badAreas': [((1026, 0), (1091, 175))]}  # Pixel areas of bad values. Each tuple pair forms a rectangle. 0:175, 1035:1091

case2 = {'case': 2,
         'dt': 1.8,
         'pressure': 1.866,
         'NPR': 4.60,
         'mag': centreMag,
         'calcNozzleExitVel': 310.5,  # m/s
         'type': 'F',
         'pos': 'C',
         'zeroRow': 360,
         'zeroCol': 58,
         'sets': [3, 4, 5],
         'badImgCutOff': 0.08,
         'h5files': ['./myfolder/sets/set1.h5', './myfolder/sets/set2.h5']}

case3 = {'case': 3,
         'dt': 2.7,
         'pressure': 1.866,
         'NPR': 4.60,
         'mag': centreMag,
         'type': 'F',
         'pos': 'C',
         'zeroRow': 360,
         'zeroCol': 58,
         'sets': [6, 7, 8, 9, 28, 30, 31, 32, 33, 34, 35],
         'badImgCutOff': 0.08,}
case4 = {'case': 4,
         'dt': 0.9,
         'pressure': 2.066,
         'NPR': 5.00,
         'mag': centreMag,
         'calcNozzleExitVel': 310.5,  # m/s
         'type': 'F',
         'zeroRow': 360,
         'zeroCol': 58,
         'pos': 'C',
         'shockReflectionPoints': [201, 340, 482, 625, 760, 890],
         'badImgCutOff': 0.08,
         'badAreas': [((1026, 0), (1091, 175))],  # Pixel areas of bad values. Each tuple pair forms a rectangle. 0:175, 1035:1091
         'sets': [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 68, 69, 70,
                  71, ]} # 72, 73, 74, 75, 76]} These last guys have wild statistics. 62, 63, 64, 65, 66, 67,also bad
case5 = {'case': 5,
         'dt': 2.7,
         'pressure': 2.066,
         'NPR': 5.00,
         'mag': centreMag,
         'calcNozzleExitVel': 310.5,  # m/s
         'type': 'F',
         'zeroRow': 360,
         'zeroCol': 58,
         'pos': 'C',
         'sets': [36, 37, 38, 39, 40, 41, 42, 43, 44, 45],
         'badImgCutOff': 0.2,}
case6 = {'case': 6,
         'dt': 1.5,
         'pressure': 1.866,
         'NPR': 4.60,
         'mag': leftMag,
         'calcNozzleExitVel': 310.5,  # m/s
         'type': 'F',
         'zeroRow': 598,
         'zeroCol': 58,
         'pos': 'L', # The L is used to indicate the camera is now focused on just the left jet.
         'sets': [83, 84, 85, 86, 87, 88],
         'badImgCutOff': 0.20,}
case7 = {'case': 7,
         'dt': 1.5,
         'pressure': 2.066,
         'NPR': 5.00,
         'mag': leftMag,
         'calcNozzleExitVel': 310.5,  # m/s
         'type': 'F',
         'zeroRow': 598,
         'zeroCol': 58,
         'pos': 'L',
         'sets': [89, 94, 95, 96, 97, 98],
         'badImgCutOff': 0.20,}
case8 = {'case': 8,
         'dt': 2.7,
         'pressure': 1.866,
         'NPR': 4.60,
         'mag': leftMag,
         'calcNozzleExitVel': 310.5,  # m/s
         'type': 'F',
         'zeroRow': 598,
         'zeroCol': 58,
         'pos': 'L',
         'sets': [80, 81, 82],
         'badImgCutOff': 0.20,}  # Set 79 lost
case9 = {'case': 9,
         'dt': 2.7,
         'pressure': 2.066,
         'NPR': 5.00,
         'mag': leftMag,
         'calcNozzleExitVel': 310.5,  # m/s
         'type': 'F',
         'zeroRow': 598,
         'zeroCol': 58,
         'pos': 'L',
         'sets': [90, 91, 92, 93],
         'badImgCutOff': 0.20,}
cases = [case1, case2, case3, case4, case5, case6, case7, case8, case9]


for case in cases:
    if False:
        pivPixelCoordsRows, pivPixelCoordsCols = getRealWorldCoords(case=case)
        case['pivPixelCoordsRows'] = pivPixelCoordsRows
        case['pivPixelCoordsCols'] = pivPixelCoordsCols

    else:
        from pivPixelCoords import pivPixelCoordsCols, pivPixelCoordsRows
        case['pivPixelCoordsRows'] = pivPixelCoordsRows
        case['pivPixelCoordsCols'] = pivPixelCoordsCols

cases = [case1, case2, case3, case4, case5, case6, case7, case8, case9]

if __name__ == '__main__':
    picklePlace = 'AlkislarStats/'
    pickle.dump(cases, open(picklePlace + 'CaseInfo.dat', 'wb'))
    print("Written case data to", picklePlace + 'CaseInfo.dat')
    print(setPath(36))