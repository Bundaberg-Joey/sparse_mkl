from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import h5py


# -----------------------------------------------------------------------------------------------------------------------------


class DataManager:
    
    def __init__(self, n):
        self.n = abs(int(n))
        self._data = [[] for _ in range(self.n)]
        
    def add_entry(self, k: int, xy: Tuple):
        xy_result = (int(xy[0]), float(xy[1]))
        self._data[k].append(xy_result)
        
    def get_X_y(self, k: int):
        xy = self._data[k]
        X = np.array([i[0] for i in xy], dtype=int).reshape(-1, 1)
        y = np.array([i[1] for i in xy], dtype=float)
        return X, y
        
    def get_all_sampled(self):
        sampled_indices = []
        for k in self._data:
            for entry in k:
                sampled_indices.append(entry[0])
        return sampled_indices
    
    def write_to_file(self, k):
        df = pd.DataFrame(data=self._data[k], columns=['mof_idx', 'performance'])
        df.to_csv(F'ami_output_process_{k}.csv', index=False)                             


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
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self[:].shape
    
    def __len__(self) -> int:
        return self.shape[0]
        
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
        
        with h5py.File(self.hdf5_loc, 'r') as f:
            X_ = f[self.data_key]
            
            if isinstance(k[0], slice):  # first index since convert to array above
                X = X_[:]  # return entire dataset
            else:
                X = np.vstack([X_[idx] for idx in k])  # avoids worries of out of order indexing / duplicate indices being passed
            
        if X.ndim == 1:
            X = X.reshape(1, -1)  # ensure always 2d output for simplicity
            
        return X    

# -----------------------------------------------------------------------------------------------------------------------------
