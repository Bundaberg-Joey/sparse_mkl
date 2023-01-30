# Sparse_mkl
* repo containing all PhD code needed for the sparse models /associated kernels and hdf5 dataclasses


## MSPARSE MODEL PROBLEMS
### The problem
* Have managed to re-write Jame's old Prospector model into something which should be agnotic for a Tanimotokernel to work which has been implemented by `GPy`
* Need to research how to implement a single kernel Tanimoto model which also has the variance as part of its kernel
* Someone did a Tanimoto kernel with variance uing GPFlow a while ago so likely a good reference
>* Worse comes to worse might be able to use GPFlow as the entire dense back end so long as correct hyperparameters can be access by the model
* Once can figure out how to do a single kernel Tanimoto, the MKL can e achieved by just fitting and applying models in a loop since the sparse model mainly just needs the covaraince matrices rather than a specific prediction ability (i.e. so can find covar using GPy but then weight outside of the model with regular python)

* [link to the GPFlow example which seems to be able to access relevant model values...](https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb)

### The Solution
* see if can easily install GPflow to a docker image with other conda packages
* then try the densemodel using GPFlow instead of GPY for an RBF model
* if the acquisition runs fine using this GPFlow model then just use GPFLow + the tanimoto kernel layout from the linked notebook above
>* benchmark costs but may aswell still use the joblib version i did since this will be running no CPU not GPU

### The goal
* By end of day Tuesday, have a tanimoto GP up and running within the sparse GP
* If time start thinking about how to do the MKL model with all of this












## TODO:
>* consider tests for  kernels (quick and only for Tanimoto and only for the actual maths)
>* need to do a screening on the COF data to make sure it doens't break (even jst doing a fit to random data would be excellent...)
>* make sure the other models don't also break
>* write the inducing functions (cluster then return centroids), consider how to do this for PCFP
>> the centroids to be the result of performing a clustering on each dataset and selectingn/4 centroids from each, final matrix to be the combination of all 4 centroids