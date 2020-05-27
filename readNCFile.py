from netCDF4 import Dataset
import numpy as np

GrahamDrive0 = '/media/graham/GrahamDrive0/TwinTestImages/Far2016/'
imagesFolder = ['Set10/']
file = 'Sequence00000000.nc'
ncFile = GrahamDrive0 + imagesFolder[0] + file

fh = Dataset(ncFile, mode='r')
#print(fh.variables)
velocity = fh.variables['velocity'][0]
pivdata = fh.variables['piv_data'][0]
pivShapeRows = pivdata.shape[0]
pivShapeCols = pivdata.shape[1]
imageCoordMax = fh.variables['piv_data'].coord_max
imageCoordMin = fh.variables['piv_data'].coord_min

pivStepx = fh.ev_IS_grid_distance_x
pivStepy = fh.ev_IS_grid_distance_y

pivPixelCoordsRows = np.arange(imageCoordMin[0], imageCoordMax[0] + pivStepy, pivStepy)
pivPixelCoordsCols = np.arange(imageCoordMin[1], imageCoordMax[1] + pivStepx, pivStepx)

print(fh.variables.keys())
import matplotlib.pyplot as plt

import matplotlib.cm as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)

plt.contourf(np.sqrt(pivdata[:,:,0]**2 + pivdata[:,:,1]**2),50)
plt.colorbar()
plt.title("Magnitude of pixel displacement")
#plt.show()

# = fh.variables