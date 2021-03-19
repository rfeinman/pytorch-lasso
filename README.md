# PyTorch-Lasso

This is a library of...

## Lasso Solvers

### Linear solvers

There are a wide variety of lasso solvers for the classical "linear" setting, i.e. the case where our dictionary is an overcomplete linear basis W. In this case we write our reconstructed signal as x_hat = Wz for a [d x k] matrix W. I have implemented a number of algorithms for this setting in `lasso.linear`. The solvers include:

- __ISTA__: Iterative shrinkage thresholding algorithms, including the "fast" variant described in [2].
- __GPSR__: The projected gradient method described in [3].
- __Interior Point__: The projected gradient method proposed in [4]. It is summarized nicely in section 2.3 of [1], and I used this description.

### Linear solvers for 2D convolution

Another "linear" setting is the case of 2D convolution. In this case, our linear basis is a large (and very sparse) matrix with block-Toeplitz structure, i.e. x_hat = conv2d(z, W). Many of the classical linear solvers are not applicable to this setting. I've implemented a few solvers specialized for the 2D convolution setting, inspired by those mentioned above:

- __ISTA__: As described above, but optimized for the conv setting. I've included code to estimate the lipschitz constant of a conv2d operator. This is needed for optimal learning rate selection.

### Nonlinear solvers

Finally, I've included some extensions for the generalized case of a nonlinear dictionary, i.e. x_hat = D(z) for a nonlinear decoder D.


## References
[1] "Least Squares Optimization with L1-Norm Regularization". Schmidt, 2005.

[2] "A Fast Iterative Shrinkage-Thresholding Algorithm". Beck & Teboulle, 2009.

[3] "Gradient projection for sparse reconstruction: Application to compressed sensing and other inverse problems". Figueiredo et al., 2007.

[4] "Atomic decomposition by basis pursuit". Chen et al., 1999.