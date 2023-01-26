# Sparse_mkl
* repo containing all PhD code needed for the sparse models /associated kernels and hdf5 dataclasses

## TODO:
>* consider tests for  kernels (quick and only for Tanimoto and only for the actual maths)
>* need to do a screening on the COF data to make sure it doens't break (even jst doing a fit to random data would be excellent...)
>* make sure the other models don't also break
>* write the inducing functions (cluster then return centroids), consider how to do this for PCFP
>> the centroids to be the result of performing a clustering on each dataset and selectingn/4 centroids from each, final matrix to be the combination of all 4 centroids