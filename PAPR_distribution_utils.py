#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to calculate PAPR distributions for evaluations in the paper
"Extreme Value Theory for Peak Power in Discrete-Time Complex Gaussian Processes"

author: Adam Wunderlich
last update: 10/28/2026
"""

import numpy as np
from scipy.stats import ecdf, bootstrap
from scipy import integrate
from scipy.optimize import brentq


def PAPR_CDF_CT(u_dB, Delta, n):
    # gumbel form
    u = 10**(u_dB/10)
    b = np.log(n) + 0.5*np.log(np.log(n)) + np.log(np.sqrt(Delta/np.pi))
    return np.exp(-np.exp(-(u-b)))

def PAPR_PDF_CT(u_dB, Delta, n):
    # gumbel form
    u = 10**(u_dB/10)
    J = np.abs(u*np.log(10)/10)
    b = np.log(n) + 0.5*np.log(np.log(n)) + np.log(np.sqrt(Delta/np.pi))
    z = u-b
    return np.exp(-(z + np.exp(-z)))*J

def PAPR_CDF_iid(u_dB, n):
    # gumbel form
    u = 10**(u_dB/10)
    b = np.log(n)
    return np.exp(-np.exp(-(u-b)))

def PAPR_PDF_iid(u_dB, n):
    # gumbel form
    u = 10**(u_dB/10)
    J = np.abs(u*np.log(10)/10)
    b = np.log(n)
    z = u-b
    return np.exp(-(z + np.exp(-z)))*J

def quantile_function(p, cdf, args = (), lb=-30, ub=30):
    if isinstance(p, float):
        cdf_minus_p = lambda x: cdf(x, *args) - p
        Q  = brentq(cdf_minus_p, lb, ub)
    else: # array argument
        Q = np.zeros(len(p))
        for i in range(len(p)):
            cdf_minus_p = lambda x: cdf(x, *args) - p[i]
            Q[i]  = brentq(cdf_minus_p, lb, ub)
    return Q
 
def PAPR_CDF_DT(u_dB, kappa, n):
    # exact discrete-time PAPR CDF under Markov exceedance model
    # scaled_Psi = Psi*(exp(-x)/(1-exp(-x)))
    u = 10**(u_dB/10)
    if kappa == 1:
        scaled_Psi = np.zeros(len(u))
    else: # kappa < 1
       def integrand(theta,u,kappa): 
           exp_arg = -u*(1-kappa**2)/(1 + 2*kappa*np.sin(theta) + (kappa**2))
           return (1 - np.exp(exp_arg))*np.exp(-u)/(2*np.pi*(1-np.exp(-u)))
        
       if isinstance(u, float):
           # there is a singularity at theta = -pi/2 when kappa=1
           scaled_Psi, _ = integrate.quad(integrand, -np.pi, np.pi, args = (u,kappa), points = -np.pi/2)
       else: # array argument
           scaled_Psi = np.zeros(len(u))
           for i in range(len(u)):
               scaled_Psi[i], _ = integrate.quad(integrand, -np.pi, np.pi, args = (u[i],kappa), points = -np.pi/2)
        
    p00 = 1 - scaled_Psi
    return (1-np.exp(-u))*p00**(n-1)
    
def W1_distance(sample, CDF, estimate_CI=True):
    """
    # estimate the W1 distance between an empirical CDF and a pre-defined CDF
    
    Parameters
    ----------
    sample : 1-D numpy array of iid samples 
    CDF : class 'function'
          CDF with parameters pre-defined.  
    estimate_CI : boolean

    Returns
    -------
    W1_dist : float
              Wasserstein-1 (Earth Mover's) distance
            
    """
    QF = lambda p: quantile_function(p, CDF, args = (), lb=-30, ub=30)      
    
    def compute_W1(data):
        res = ecdf(data)
        CDF_emp = res.cdf.evaluate
        QF_emp = lambda p: quantile_function(p, CDF_emp, args = (), lb=-30, ub=30)
        p_lower = 0.001
        p_upper = 0.999
        a = min(QF(p_lower), QF_emp(p_lower))
        b = max(QF(p_upper), QF_emp(p_upper))
        integrand = lambda u: np.abs(CDF_emp(u) - CDF(u))
        I, _ = integrate.quad(integrand, a, b, epsabs=1e-3, epsrel=1e-3, limit=1000)
        return I
      
    W1_dist = compute_W1(sample)
    if estimate_CI:  # estimate 95% bootstrap CI (percentile method)
        res = bootstrap((sample,), compute_W1, n_resamples=100, vectorized=False, confidence_level=0.95, method='percentile')
        CI = np.array([res.confidence_interval[0], res.confidence_interval[1]])
    else:
        CI = None
    return W1_dist, CI

def kappa_hat(Rzz):
    # given a complex-valued ACF from 0 lag to Rzz_length-1,
    # returns kappa hat
    kappa = np.max(np.abs(Rzz[1:]))
    
    return kappa

def spectral_parameters(B1, fc1, B2, fc2):
    def sinc(x):
        if x==0:
            return 1
        else:
            return np.sin(np.pi*x)/(np.pi*x)
          
    Rzz_length = 10
    Rzz = np.ones(Rzz_length, dtype=complex)
    for k in np.arange(Rzz_length):
        if k > 0:
            Rzz[k] = (B1*np.exp(1j*2*np.pi*fc1*k)*sinc(B1*k) + B2*np.exp(1j*2*np.pi*fc2*k)*sinc(B2*k))/(B1+B2)
    kappa = kappa_hat(Rzz) 
    Delta = ((np.pi**2)/(3*(B1+B2)**2))*((B1**3 + B2**3)*(B1+B2) + 12*B1*B2*(fc1-fc2)**2)

    return Delta, kappa