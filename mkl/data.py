import numpy as np
from numpy.typing import NDArray
import h5py


# -----------------------------------------------------------------------------------------------------------------------------


class Hdf5Dataset:
    """hdf5 dataset containing features for MOFs.
    Loads features from HDF5 dataset for passed indices.
    Useful for scenarios where multiple kernels in one model each using different feature sets.
    """
    
    def __init__(self, hdf5_loc: str, data_key: str='X') -> None:
        self.hdf5_loc = str(hdf5_loc)
        self.data_key = str(data_key)
        
    def __repr__(self) -> str:
        return F'KernelDataset(hdf5_loc="{self.hdf5_loc}", data_key="{self.data_key}")'
        
    def __getitem__(self, k: NDArray[np.int_]) -> NDArray[NDArray]:
        """return features from hdf5 dataset for passed indices.

        Parameters
        ----------
        k : NDArray[np.int_]
            indices to load data from 

        Returns
        -------
        NDArray[NDArray]
            feature matrix where each row are the features of the specified index.
        """
        k = np.asarray(k).ravel()
        
        # dual argsorts as need to pass indices to hdf5 in increasing order
        # but need to return w.r.t passed indices
        ka = np.argsort(k)
        kaa = np.argsort(ka)
        
        with h5py.File(self.hdf5_loc, 'r') as f:
            X_ = f[self.data_key]
            X = X_[k[ka]][kaa]
            
        return X
    

# -----------------------------------------------------------------------------------------------------------------------------
