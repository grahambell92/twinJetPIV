%% Script to load the modules for PIV processing on Massive.


#Setup python virtualenv - need to use this path to avoid 'module load'
/usr/local/python/3.6.2-static/bin/python3 -m venv .

#Activate the virtual environment - do this for each shell session
source ./bin/activate

#Upgrade pip
pip install --upgrade pip
pip install numpy
pip install netCDF4
pip install h5py
pip install dask
pip install toolz
pip install scipy
pip install matplotlib