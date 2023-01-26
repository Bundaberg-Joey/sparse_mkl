import numpy as np
import h5py


class KernelDataset:
    
    def __init__(self, hdf5_loc, data_key='X'):
        self.hdf5_loc = str(hdf5_loc)
        self.data_key = str(data_key)
        
    def __repr__(self):
        return F'KernelDataset(hdf5_loc="{self.hdf5_loc}", data_key="{self.data_key}")'
        
    def __getitem__(self, k):
        k = np.asarray(k)
        ka = np.argsort(k)
        kaa = np.argsort(ka)
        
        with h5py.File(self.hdf5_loc, 'r') as f:
            X_ = f[self.data_key]
            X = X_[k[ka]][kaa]
            
        return X