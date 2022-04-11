# Overview
This repository implments a pure-Python/Numpy version of Kalman filtering with additional functionality for conjugate sampling of evolution and observation variances. Additionally, it includes functionality for integration with `pymc3` allowing for *composite filtering and Hamiltonian Monte Carlo* for very large dynamic problems involving non-Gaussian data or extensive hierarchical model structure. Another set of utilities provide functionality for grid search over model hyperparameters.

The core routines in this repository are built around the Bayesian filtering and forecasting theory from the book by Prado, Ferreira & West 2021
titled Time Series: Modeling, Computation & Inference. This theory builds upon a rich tradition of filtering for Gaussian linear dynamical systems laid down by Kalman, Thiele, and many others.

For an example of how this works, you can check out the sample notebook included to see how the filtering logic can be used for a dynamic Weibull extreme model.


