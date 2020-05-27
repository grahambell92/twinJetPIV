# PIV Processing Codes
## Graham Bell, 2016 PIV Supersonic twin-jet dataset

This repo contains the basic PIV processing codebase used by Graham Bell during his PhD 2015-2020.
Graham Bell was supervised by Dr. Daniel Edgington-Mitchell and Prof. Damon Honnery.

The application of this codebase was on a PIV analysis of s/D=3.0 twin jets at NPR=4.6 and 5.0.
You can read the case details in PIVCasesInfo.py


## Experimental Case Overview
A presentation of the dataset is contained within the publication:
Bell, G., Soria, J., Honnery, D. et al. An experimental investigation of coupled underexpanded supersonic twin-jets. Exp Fluids 59, 139 (2018). https://doi.org/10.1007/s00348-018-2593-1

NPR=4.6 and 5.0 were examined at a fixed twin-jet spacing of s/D=3.
The best cases are case1 (NPR=4.6) and case4 (5.0).
The dataset is comprised of 9 cases consisting of about 6TB of image snapshots.

The other cases consist of experimentation with different camera positioning and interframe timing:
centreMag: is the magnification for when the camera was placed in line with the symmetry plane of the jets.
leftMag: is the magnification for when the camera was placed examining the left jet and its farfield.
900 ns was found to be an adequate interframe timing to produce acceptable velocities that fit within the dynamic range within the PIV processing algorithm.
Longer inter frame times of 1.8 and 2.7 us were included in cases like 3 and 5 to increase the displacements in the shear layer and farfield.
These larger displacements mean that the jet core is not resolvable. These cases were never explored within GB's PhD.




## General Codebase Workflow
### Images:
The Imperx camera takes images with 12 bit resolution in a 16 bit .tiff format. The images are approx 54MB each.
The camera takes double shutter images - two frames separated by ~800us. These form PIV image pairs.
The camera takes these images pairs at 0.5Hz, and can store 1000 in memory.
The 1000 images are called a set and recorded in a folder by set number.
Multiple sets are used to form the images for a particular jet operating condition, called a case.
The case folders and jet properties are recorded by PIVCasesInfo.py
The large file size of the images meant that the sets needed to be split over serveral disks, called GrahamDrive0, 1, 2, 3.

### Image preparation:
PIV Processing and Vector fieldsThe images recorded from the camera are encoded from 0 to 2^12 within the tiff file, but the tiff file uses a 2**16 bit container.
Therefore the image values need to be interpolated such that they cover the 0-2^16 value range.
Additionally, the illumination across the image can be improved and made more uniform, which improves the PIV correlation result.
Lastly, the images from each set can be median subtracted to improve the PIV correlation.
These steps are performed by PIVImagePreProcessing.py

### PIV processing:
PIVView is a commercial software used for calculating vector displacements in image pairs.
LTRAC has (had) a licence for PIVView and it was used to compute the vector displacements.
PIVView can be run in both with a GUI and in a command line mode.
The GUI is used to get the PIV processing parameters just right, and these parameters are saved to a .par file.
The command line tool (called dpiv) takes a .par file and image paths and computes the displacement field (in pixels).
The displacement fields are stored in NC files.
dpiv computes only one image pair at a time. A routine was written to farm a job list of dpiv image pair commands to multi-processing scheduler, so that large datasets are computed in parallel.
Managing the case and file logistics, as well as computing the dpiv program in parallel is performed by processPIVParallel.py
The NC files are for single image pairs and are picked up and compiled into an H5 array by processPIVParallel.py
The original images and their background subtracted counterparts are compressed and stored in .tar.bz2 format as it had the best file reduction and write speed. This is also performed by processPIVParallel.py.

### Data validation:
Various different types of data validation can be performed.
Within GB's PhD, a temporal outlier filter (Chauvenet) and hole repair algorithm are implemented.
These validations and repairs are performed directly on the H5 vector arrays for storage size restraints. This should only be performed once.
temporalValidator.py performs this validation.

### Basic Statistical Processing:
Reading in a single set (approx 500 vector fields) occupies about 8GB of memory.
Therefore it is typically not possible to perform casewise statistical analysis in RAM, except for when using a computing service with significant RAM like MASSIVE.
An out-of-core computational python library called Dask was used during GB's PhD to compute casewise statistics.
Dask arrays are similar to numpy arrays, but there are some adaptions.
All of the generation of out-of-core arrays and basic statistics are performed by the dask library: daskPIVStats.py
pivPixelCoords.py It became frustrating to convert the array indexes to real world coordinates. This file stores case1 and case4 coordinates in m.
The vector fields are stored in pixel displacements and must be multiplied by the magnification factor and inter-frame time to convert them to velocity.


### POD Modal analysis:
Some of the linear algebra required to perform Proper Orthogonal Decomposition (POD) was not available in Dask.
Many alternatives were tried to do out-of-core computations of POD, including an implementation of the modred python package.
It was required to have the entire case loaded into memory to compute the POD. This meant that all POD calculations had to be done on the MASSIVE service.
A python class was written to neatly handle the case structure, called PODClass.py.
And an example file on how to compute the in-memory POD on this dataset is included in POD_inMem_classOriented.py

### Working with MASSIVE:
Several scripts have been included that help working with MASSIVE when using this dataset.
copyH5ToMassiveServer.py helps copy the H5 files and directory structure to the remote MASSIVE file system.
Python3.6 is not installed on Massive. GB has spoken with MASSIVE staff and they say it is on their todo list but it is not as simple as a it sounds.
Therefore, installing python3.6 and the associated python libraries is performed on the user account by massive_loadModules.sh
The best workflow for using MASSIVE for POD and your local computer with Dask was to implement a git repo on both machines and use if statements to detect whether the MASSIVE settings should be used within python scripts.


