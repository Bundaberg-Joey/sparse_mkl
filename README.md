# Sparse_mkl
* repo containing all PhD code needed for the sparse models /associated kernels and hdf5 dataclasses


## MSPARSE MODEL PROBLEMS
### The problem
* Sparse GP is only returning the mean of the passed training data when asked to make a prediction
* There is some small fluctuation in the returned values but only in the 1st or 2nd decimal place
* This is also true for predictions on data points already seen by the model (which would expect to see just the training values returned...)
* Because the model can run from end to end quickly for large numbers of samples, it is likely that this is NOT a matrix / shape issue but instead a parameter is not being inferred or set correctly which then causes down stream issues.
* For the moment, the likely place to investigate is the hyperparameters extrated from the internal model (previously GPy but not sklearn)
* If that still does not solve the problem then either a different cause is responsible or it is a joint responsibility beteen problems of param extraction AND application

### Current Goal
* Get the sparse GP model to behave "sensibly" for the RBF kernel (wither GPy or sklearn)
* Once that's done the next goal will be to get a Tanimoto kernel to work
* After that, the next goal will be getting the mkl model to work for multiple tanimoto kernels

### Steps to Take for the sparse GP model to behave with RBF
#### 1. Establish a concrete dataset for evaluating this problem
* helps between tests / keeps everything "lined up"
* small number of features / samples etc for speed

#### 2. change sparse GP to use the GPy code internally 
* see if this can solve the problem from the outset
* can worry about applying kernel directly etc after

#### 2. Asses performance of "original" sparse model by James on this dataset
* lets you know how things should behave when working


















## TODO:
>* consider tests for  kernels (quick and only for Tanimoto and only for the actual maths)
>* need to do a screening on the COF data to make sure it doens't break (even jst doing a fit to random data would be excellent...)
>* make sure the other models don't also break
>* write the inducing functions (cluster then return centroids), consider how to do this for PCFP
>> the centroids to be the result of performing a clustering on each dataset and selectingn/4 centroids from each, final matrix to be the combination of all 4 centroids