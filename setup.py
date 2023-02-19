from setuptools import setup, find_packages

setup(
    name='pytorch-lasso',
    version='0.0.1',
    url='https://github.com/epurdy/pytorch-lasso.git',
    author='Reuben Feinman',
    author_email='',
    description='L1-regularized least squares with PyTorch',
    packages=find_packages(),    
    install_requires=[
      'torch >= 1.13.0'
    ],
)
