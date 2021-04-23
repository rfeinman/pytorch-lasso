# PyTorch Lasso

A PyTorch library for L1-regularized least squares (lasso) problems. This library is a work in progress; contributions are welcome and appreciated!

Author: Reuben Feinman (New York University)

__At a glance:__

```python
import torch
from lasso.linear import dict_learning, sparse_encode

# dummy data matrix
data = torch.randn(100, 10)

# Dictionary Learning
dictionary, losses = dict_learning(data, n_components=50, alpha=0.5, algorithm='ista')

# Sparse Coding (lasso solve)
coeffs = sparse_encode(data, dictionary, alpha=0.2, algorithm='interior-point')
```

__Lasso solvers:__ ISTA, Gradient Projection, Interior Point, Iterative Ridge, Coordinate Descent, Split Bregman

__Examples:__ See the [Omniglot tutorial](https://github.com/rfeinman/pytorch-lasso/blob/master/examples/dict_learning_omniglot.ipynb) for a demonstration of how to use pytorch-lasso to perform dictionary learning with image patches.

## 1. Overview

Pytorch-lasso is a collection of utilities for sparse coding and dictionary learning in PyTorch. 
The aim of the project is few-fold: 1) to assemble a collection of classical sparse coding techniques for benchmarking and comparison, 2) to provide a modern implementation of popular algorithms with autograd and GPU support, and 3) to offer a starting point for nonlinear sparse coding problems where gradient and/or hessian information is available.

The canonical lasso formulation is an L1-regularized (linear) least squares problem with the following form:

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\begin{align*}\min_{z}&space;\frac{1}{2}&space;||&space;Wz&space;-&space;x&space;||_2^2&space;&plus;&space;\alpha&space;||z||_1\end{align*}&space;" title="http://latex.codecogs.com/svg.latex?\begin{align*}\min_{z} \frac{1}{2} || Wz - x ||_2^2 + \alpha ||z||_1\end{align*} " /> 
</p>

where <img src="http://latex.codecogs.com/svg.latex?x&space;\in&space;\mathbb{R}^d" title="http://latex.codecogs.com/svg.latex?x \in \mathbb{R}^d" /> is an observation vector, <img src="http://latex.codecogs.com/svg.latex?W&space;\in&space;\mathbb{R}^{d&space;\times&space;k}" title="http://latex.codecogs.com/svg.latex?W \in \mathbb{R}^{d \times k}" /> a dictionary "weight" matrix, and <img src="http://latex.codecogs.com/svg.latex?z&space;\in&space;\mathbb{R}^k" title="http://latex.codecogs.com/svg.latex?z \in \mathbb{R}^k" /> a vector of sparse coefficients. 
Typically the dictionary is overcomplete, i.e. <img src="http://latex.codecogs.com/svg.latex?k&space;>&space;d" />.
Pytorch-lasso includes a number of techniques for solving the linear lasso problem, detailed in Section 2. 
I'm grateful to Mark Schmidt, whose lecture notes guided my literature review [1].

In addition to solving for sparse coefficients with an existing dictionary, another problem of interest is _dictionary learning_. 
Dictionary learning is a matrix factorization problem formulated as follows:

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\begin{align*}\min_{W,Z}&space;\frac{1}{2}&space;||&space;ZW^T&space;-&space;X&space;||_2^2&space;&plus;&space;\alpha&space;||Z||_1\end{align*}&space;" title="http://latex.codecogs.com/svg.latex?\begin{align*}\min_{W,Z} \frac{1}{2} || ZW^T - X ||_2^2 + \alpha ||Z||_1\end{align*} " />
</p>

In this case, <img src="http://latex.codecogs.com/svg.latex?X&space;\in&space;\mathbb{R}^{n&space;\times&space;d}" title="http://latex.codecogs.com/svg.latex?X \in \mathbb{R}^{n \times d}" /> and <img src="http://latex.codecogs.com/svg.latex?Z&space;\in&space;\mathbb{R}^{n&space;\times&space;k}" title="http://latex.codecogs.com/svg.latex?Z \in \mathbb{R}^{n \times k}" /> are observation and coefficient _matrices_ with n samples. 
The problem is typically solved in an EM fashion by iterating between finding the optimal coefficients (lasso) for the current dictionary and finding the optimal dictionary (least-squares) for the current coefficients.
Pytorch-lasso includes modules for dictionary learning in two forms: 1) a "constrained" setting where dictionary atoms are constrained to unit norm (a la scikit-learn), and 2) an "unconstrained" setting where the unit constraint is replaced by an L2 dictionary penalty. Details are provided in Section 3.

## 2. Lasso Solvers

### Linear

There are a variety of lasso solvers for the classical "linear" setting, i.e. the case where our dictionary is an overcomplete linear basis W.
The `lasso.linear` module gathers a number of popular algorithms for this setting. The solvers include:

- __ISTA__: Iterative shrinkage thresholding algorithms, including the "fast" variant described in [2]. An option for backtracking line-search is provided. This algorithm is very efficient and produces good results in most cases. It's a great default to use.
- __Gradient Projection__: The projected gradient method described in [3]. The lasso problem is reformulated as a box-constrained quadratic programming problem and solved by a gradient projection algorithm. At the moment, only the _GPSR-Basic_ variant of the algorithm is offered.
- __Interior Point__: The primal-dual log barrier (interior point) method proposed in [4]. Also known as _basis pursuit_. The algorithm is summarized nicely in [5], and also in section 2.3 of [1]. These summaries informed the current implementation.
- __Iterative Ridge__: An iterative algorithm developed in [6] that uses an update rule inspired by ridge regression, along with the approximation <img src="http://latex.codecogs.com/svg.latex?||z||_1&space;\approx&space;||z||_2^2&space;/&space;||z||_1" />. The updates are applied iteratively since the step depends on z (unlike classical ridge). An optional line search can make convergence much faster. Although less efficient than ISTA/CD, this algorithm consistently produces the best results in my evaluations thus far. Runtime on GPU is very reasonable.
- __Coordinate Descent__: A popular approach for sparse coding developed in [7]; often considered the fastest algorithm for sparse code inference. The current cd implementation is a batched variant of the per-sample CD algorithms offered in scikit-learn. It does not always work as expected; there may be some bugs. I've included a "modified" variant (coord_descent_mod) derived from sklearn that is much slower but produces reliable results.
- __Split Bregman__: The Split Bregman method of [8].

### 2D Convolution

Another "linear" setting of interest is 2D convolution. 
In this case, our linear basis is a large (and very sparse) matrix with block-Toeplitz structure. 
Many of the classical linear solvers are not applicable to this setting. 
The `lasso.conv2d` module implements a few solvers specialized for the 2D convolution setting, inspired by those discussed above. They are:

- __ISTA__: As described above, but optimized for the conv setting. I've included code to estimate the lipschitz constant of a conv2d operator. This is needed for optimal learning rate selection.
- __Gradient Projection__: This algorithm works with convolution, but is not yet included by default in the `lasso.conv2d` library. It is coming soon.

### Nonlinear

The module `lasso.nonlinear` contains some extensions for the generalized case of a nonlinear dictionary, i.e. x_hat = D(z), where D is a nonlinear decoder. 
This module is experimental: with the exception of ISTA and OWLQN--which are established techniques for nonlinear lasso--the solvers here represent my own ideas for extending linear methods.

The current offerings are as follows:
- __ISTA__: A generalized ISTA solver for nonlinear decoders. I've included a "dynamic" learning rate option which estimates the lipschitz constant of the RSS gradient function by linearizing this function around the current point.
- __OWLQN__: The Orthant-wise limited memory quasi-Newton (OWLQN) method presented in [9]. This is a popular technique for solving nonlinear lasso problems. 
- __Iterative Ridge BFGS__: A generalization of the Iterative Ridge method. Uses Newton steps with a BFGS approximation of the RSS hessian.
- __Split Bregman__: A generalization of the Split Bregman algorithm. This algorithm uses the Gauss-Newton method, a modification of Newton's method specialized for least squares objective terms.


## 3. Dictionary Learning

In addition to lasso solvers, pytorch-lasso provides implementations of dictionary learning (sparse matrix factorization). 
The structure is inspired by Scikit Learn's `sklearn.decomposition` module. 
In sklearn, the dictionary is constrained such that each component vector has unit norm. 
This framework is great for linear models, however, it's unclear how it extends to nonlinear decoders. 
Pytorch-lasso offers two variants of the dictionary learning problem: 1) the "constrained" unit-norm variant, and 2) an "unconstrained" counterpart with L2 dictionary regularization.


## References
[1] "Least Squares Optimization with L1-Norm Regularization". Schmidt, 2005.

[2] "A Fast Iterative Shrinkage-Thresholding Algorithm". Beck & Teboulle, 2009.

[3] "Gradient projection for sparse reconstruction: Application to compressed sensing and other inverse problems". Figueiredo et al., 2007.

[4] "Atomic decomposition by basis pursuit". Chen et al., 1999.

[5] "Block coordinate relaxation methods for nonparametrix wavelet denoising". Sardy et al., 2000.

[6] "Variable selection via non-concave penalized likelihood and its oracle properties". Fan and Li, 2001.

[7] "Coordinate descent optimization for L1 minimization with application to compressed sensing; a greedy algorithm". Li and Osher, 2009.

[8] "The split Bregman method for L1-regularized problems". Goldstein & Osher, 2008.

[9] "Scalable training of L1-regularized log-linear models". Andrew & Gao, 2007.