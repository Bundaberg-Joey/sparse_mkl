# Sparse_mkl
* repo containing all PhD code needed for the sparse models /associated kernels and hdf5 dataclasses

## TODO:
* determine the inducing matrices for each feature (physical, PCFP_0, PCFP_1, PCFP_2) -> can use KMeans for physical but use a different clustering for the PCFP approaches

* RBF model has one set of inducing points (whatever largest amount permitted is from Prospector)
>>* MKL model to have inducing points sampled by clustering each set of fingerprints and sampling max_ize/3 from each (3 because 3 kernels in MKL)


* create HDF5 datasets for the test screenings and confirm everything still works ok

* Message gael saying should be good to go whenever he is