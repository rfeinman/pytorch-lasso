# PyTorch Lasso

Author: Reuben Feinman (New York University)

## Overview

Pytorch-lasso is a collection of utilities for sparse coding and dictionary learning in PyTorch. The aim of the project is few-fold: 1) assemble a collection of different sparse coding techniques from the literature to support benchmarking and comparisons, 2) provide a modern implementation of popular algorithms with autograd and GPU support.

Lasso is an L1-regularized least squares problem with the following form:

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\begin{align*}\min_{z}&space;\frac{1}{2}&space;||&space;Wz&space;-&space;x&space;||_2^2&space;&plus;&space;\alpha&space;||z||_1\end{align*}&space;" title="http://latex.codecogs.com/svg.latex?\begin{align*}\min_{z} \frac{1}{2} || Wz - x ||_2^2 + \alpha ||z||_1\end{align*} " /> 
</p>

## Lasso Solvers

### Linear solvers

There are a wide variety of lasso solvers for the classical "linear" setting, i.e. the case where our dictionary is an overcomplete linear basis W. In this case we write our reconstructed signal as x_hat = Wz for a [d x k] matrix W. I have implemented a number of algorithms for this setting in `lasso.linear`. The solvers include:

- __ISTA__: Iterative shrinkage thresholding algorithms, including the "fast" variant described in [2]. This algorithm is very efficient and produces good results in most cases. It's a good default.
- __GPSR__: The projected gradient method described in [3]. The lasso problem is reformulated as a box-constrained quadratic programming problem and solved by a gradient projection algorithm. At the moment, I've only implemented the _GPSR-Basic_ variant of the algorithm.
- __Interior Point__: The primal-dual interior point method proposed in [4] (also known as _basis pursuit_). The algorithm is summarized nicely in [5], and also in section 2.3 of [1]. I used these descriptions to develop my implementation.
- __Iterative Ridge__: An iterative approach developed by [6]. Using the approximation norm(z, 1) = norm(z, 2)^2 / norm(z, 1), this method applies an update rule inspired by ridge regression. The updates are applied iteratively since the step now depends on z. I've included an optional line search (used by default) that makes convergence much faster. This method is the fastest and most consistent in my experiments.
- __Coordinate Descent__: A classical approach for sparse coding (often thought as the fastest algorithm). My current implementation is a work-in-progress.

### Linear solvers for 2D convolution

Another "linear" setting is the case of 2D convolution. In this case, our linear basis is a large (and very sparse) matrix with block-Toeplitz structure, i.e. x_hat = conv2d(z, W). Many of the classical linear solvers are not applicable to this setting. I've implemented a few solvers specialized for the 2D convolution setting, inspired by those above. They are:

- __ISTA__: As described above, but optimized for the conv setting. I've included code to estimate the lipschitz constant of a conv2d operator. This is needed for optimal learning rate selection.

### Nonlinear solvers

Finally, I've included some extensions for the generalized case of a nonlinear dictionary, i.e. x_hat = D(z) for a nonlinear decoder D.


## References
[1] "Least Squares Optimization with L1-Norm Regularization". Schmidt, 2005.

[2] "A Fast Iterative Shrinkage-Thresholding Algorithm". Beck & Teboulle, 2009.

[3] "Gradient projection for sparse reconstruction: Application to compressed sensing and other inverse problems". Figueiredo et al., 2007.

[4] "Atomic decomposition by basis pursuit". Chen et al., 1999.

[5] "Block coordinate relaxation methods for nonparametrix wavelet denoising". Sardy et al., 2000.

[6] "Variable selection via non-concave penalized likelihood and its oracle properties". Fan and Li, 2001.