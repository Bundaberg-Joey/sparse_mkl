# Sparse_mkl
* repo containing all PhD code needed for the sparse models /associated kernels and hdf5 dataclasses


## Work
### Todo
* ~~rework james' old model to something more agnostic~~
* ~~use gpflow as the backend instead of GPy~~
* implement tanimoto kernel in GPFlow  
* implement MKL using manual weighted covariance combination

### Tasks
1. implement tanimoto kernel from link [link to the GPFlow example which seems to be able to access relevant model values...](https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb)
2. Try to use joblib with it (though potentially not likely)
3. see if can associate index values / dataset with it though no worries if not

### The goal
* By end of day Tuesday, have a tanimoto GP up and running within the sparse GP
* If time start thinking about how to do the MKL model with all of this












## TODO:
>* consider tests for  kernels (quick and only for Tanimoto and only for the actual maths)
>* need to do a screening on the COF data to make sure it doens't break (even jst doing a fit to random data would be excellent...)
>* make sure the other models don't also break
>* write the inducing functions (cluster then return centroids), consider how to do this for PCFP
>> the centroids to be the result of performing a clustering on each dataset and selectingn/4 centroids from each, final matrix to be the combination of all 4 centroids