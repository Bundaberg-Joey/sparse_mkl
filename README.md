# Sparse_mkl
* repo containing all PhD code needed for the sparse models /associated kernels and hdf5 dataclasses


## Work
### Todo
* Need to have MKL be able to update / select its own weights 



## TODO:
>* consider tests for  kernels (quick and only for Tanimoto and only for the actual maths)
>* need to do a screening on the COF data to make sure it doens't break (even jst doing a fit to random data would be excellent...)
>* make sure the other models don't also break
>* write the inducing functions (cluster then return centroids), consider how to do this for PCFP
>* RBF model has one set of inducing points (whatever largest amount permitted is from Prospector)
>>* MKL model to have inducing points sampled by clustering each set of fingerprints and sampling max_ize/3 from each (3 because 3 kernels in MKL)