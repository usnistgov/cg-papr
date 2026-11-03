# <u>**cg-papr: Peak Power Distributions of Discrete-Time Complex Gaussian Processes**</u>

## Overview
This repository contains Python code to implement simulations and evaluations for peak power distributions of discrete-time Complex Gaussian processes presented in the paper cited below.  

## Reference
A. Wunderlich, "Extreme Value Theory for Peak Power in Discrete-Time Complex Gaussian Processes", preprint 2026.  

## Getting Started
Create a Conda environment using the command ```conda env create -f environment.yml```.  

Three scripts can be used to reproduce the plots in the paper:

- ``monte_carlo_simulations.py`` 
  - Monte Carlo simulations of peak-to-average power (PAPR) distributions for filtered complex white Gaussian noise.  Saves a .json data file.

- ``eval_W1_dists.py`` 
  - Compute Wasserstein-1 (W1) distances between empirical PAPR distribution and analytical models.  Loads precomputed .json file and adds W1 distances.  

- ``make_plots.py`` 
  - Reproduce plots provided in the above paper.  Loads precomputed .json file.
 
  The data\ directory contains the ``.json`` data files corresponding to the figures in the paper.  

