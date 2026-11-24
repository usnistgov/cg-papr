# <u>**cg-papr: Peak Power Distributions of Discrete-Time Complex Gaussian Processes**</u>

## Overview
This repository contains Python code to implement simulations and evaluations for peak power distributions of discrete-time Complex Gaussian processes presented in the paper cited below.  

## Reference
A. Wunderlich, "Extreme Value Theory for Peak Power in Discrete-Time Complex Gaussian Processes", [arXiv preprint:2611.29820](https://doi.org/10.48550/arXiv.2611.29820), 2026.

## Getting Started
The Python environment used to run the code can be recreated with the command ``conda env create -f environment.yml``.  

The ``data\`` directory contains ``.json`` data files corresponding to the Monte Carlo simulations presented in the paper.  

The ``src\`` directory contains source code, including three scripts:

- ``monte_carlo_simulations.py`` 
  - Monte Carlo simulations of peak-to-average power (PAPR) distributions for filtered complex white Gaussian noise.  Saves a ``.json`` data file like those in ``data\``.  

- ``eval_W1_dists.py`` 
  - Computes Wasserstein-1 (W1) distances between empirical PAPR distribution and analytical models.  Loads precomputed ``.json`` file and adds W1 distances to the file.  

- ``make_plots.py`` 
  - Loads precomputed ``.json`` file, like those stored in ``data\``.  Reproduces plots provided in the above paper.  
 


