# -*- coding: utf-8 -*-
"""
Monte Carlo simulation script for evaluations in the paper
"Extreme Value Theory for Peak Power in Discrete-Time Complex Gaussian Processes"

author: Adam Wunderlich
last update: 11/18/2026

"""

import time
import json
import numpy as np
import scipy.signal as signal
from numpy.random import default_rng
from json import JSONEncoder
from multiprocessing import Process, Queue


class NumpyArrayEncoder(JSONEncoder):  # custom json encoder for numpy arrays
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


rng = default_rng()  # random number generator object

def FWN_MC_trial(queue, n, i, sos1, sos2):
    '''
    Compute single Monte Carlo trial for filtered white noise simulations.

    Parameters
    ----------
    queue : Queue object from multiprocessing module
        Used to pass results from each process.
    n : sample size
    i : MC trial index.
    sos1 : IIR sos filter coefficients for component 1
    sos2 : IIR sos filter coefficients for component 2

    Returns
    -------
    None.

    '''

    rng = np.random.default_rng() # initialize new instance of random number generator
    pad = 2**12
    I = rng.normal(0, 1, n + 2*pad)
    Q = rng.normal(0, 1, n + 2*pad)
    white_noise = I + 1j*Q
    filtered_WN1 = signal.sosfilt(sos1, white_noise)
    if sos2 is None:
        filtered_WN = filtered_WN1
    else:
        filtered_WN2 = signal.sosfilt(sos2, white_noise)
        filtered_WN = filtered_WN1 + filtered_WN2

    # keep middle n samples to avoid transients on ends
    x = filtered_WN.real[pad:(pad+n)]
    y = filtered_WN.imag[pad:(pad+n)]
    r_sq = x**2 + y**2

    Pmax = np.max(r_sq)
    Pmean = np.mean(r_sq)
    PAPR = Pmax/Pmean
    queue.put((i, PAPR))

def compute_PAPR_filtered_WGN_MC_parallel(n, num_MC, filt_sos1, filt_sos2=None, batch_size=1000):
    '''
    Multiprocessing parallel implementation of Monte Carlo PAPR experiments
    with IQ imbalance

    Parameters
    ----------
    n : int
        length of discrete-time signal
    num_MC : int
        Number of Monte Carlo trials
    filt_sos1 : numpy array
        Complex sos filter coefficients for component 1
    filt_sos2 : numpy array
        Complex sos filter coefficients for component 2
    batch_size : int, optional
        Batch size for parallel processing. Must divide evenly into num_MC.

    Returns
    -------
    PAPR : 1-D numpy array of length num_MC
        Monte Carlo samples of PAPR

    '''

    assert num_MC % batch_size == 0
    q = Queue()
    PAPR = np.zeros(num_MC)
    for i in range(0, num_MC, batch_size):
        # execute all trials in a batch
        processes = [Process(target=FWN_MC_trial, args=(q, n, j, filt_sos1, filt_sos2)) for j in range(i, i+batch_size)]
        # start all processes
        for process in processes:
            process.start()
        for process in processes:
            ret = q.get()
            PAPR[ret[0]] = ret[1]
        # wait for all processes to complete
        for process in processes:
            process.join()
    return (PAPR)

def design_iir_filter(filter_order, symmetric, fc, B, gain=1):
    """
    Design iir filter in second-order sections (SOS) form

    Parameters
    ----------
    filter_order : int
        Filter order
    symmetric : boolean
        Specifies if filter passband is symmetric wrt zero frequency
    fc : float
        Passband center frequency specified in range [-0.5,0.5] cycles per sample
    B : float
        Passband bandwidth specified in range [0, 0.5] cycles per sample
    gain : float, optional
        Filter gain. The default is 1.

    Returns
    -------
    sos : numpy array
        SOS filter coefficients.
    """

    # Note: signal.iirfilter expects cutoffs in range [0, 1.0] half-cycles per sample


    if symmetric: # real-valued filter with symmetric frequency response around zero frequency
        if fc < 0:
            fc = -fc
        fl = fc-B/2  # lower cutoff
        fu = fc+B/2  # upper cutoff
        if fl < 0:
            fl = 0
        if fl == 0: # LP
            sos = signal.iirfilter(filter_order, 2*fu, btype = 'lowpass', ftype = 'butter', output='sos')
        elif fu == 0.5: # HP
            sos = signal.iirfilter(filter_order, 2*fl, btype = 'highpass', ftype = 'butter', output='sos')
        else: # BP
            sos = signal.iirfilter(filter_order, [2*fl, 2*fu], btype = 'bandpass', ftype = 'butter', output='sos')
    else: # complex-valued filter with asymmetric frequency response around zero frequency
        # design LP filter with desired 2-sided BW and phase shift sos coefficients
        # reference: Crystal, T., and L. Ehrman. "The design and applications
        # of digital filters with complex coefficients", IEEE Transactions on Audio and Electroacoustics, 1968.

        # real LP - cutoff specified in range [0,1] (half-cycles per sample)
        sos_LP = signal.iirfilter(filter_order, B, btype = 'lowpass', ftype = 'butter', output='sos')
        omega_c = 2*np.pi*fc
        W = np.exp(1j*omega_c)
        sos = sos_LP.astype(complex)
        for i in range(sos.shape[0]):
          # transform each sos filter with freq response
          # H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
          sos[i,1] *= W    # b1
          sos[i,2] *= W**2 # b2
          sos[i,4] *= W    # a1
          sos[i,5] *= W**2 # a2

    sos[0, :3] *= gain # multiply numerator coefficients of first second-order section by desired gain

    return sos

#%%
def main():

    #%%
    # filtered WGN Monte Carlo
    num_MC =  int(1e5) # number of MC trials per condition
    scenario = 'dual-band_asymmetric' #'single-band_symmetric', 'dual-band_symmetric', 'dual-band_asymmetric'
    fname = scenario  + f'-{num_MC:.0e}'.replace('+','') + '.json'

    # Note: half-bandwidths below 0.02 are hard to approximate with a digitial filter

    if scenario == 'single-band_symmetric':
        symmetric = True
        OSR_L = np.array([1.25, 2, 4, 8]) # Landau oversampling ratio
        OSR_N = np.array([1.25, 2, 4, 8]) # Nyquist oversampling ratio
        B = 1/(2*OSR_L) # half-bandwidth
        B1 = 2*B
        B2 = np.zeros(len(OSR_L))
        fc1 = np.zeros(len(OSR_L))
        fc2 = np.zeros(len(OSR_L))
    elif scenario == 'dual-band_symmetric':
        symmetric = True
        OSR_L = np.array([2, 4, 8, 16])
        OSR_N = np.ones(4)*1.25
        B = 1/(2*OSR_L)  # half-bandwidth
        B1 = B
        B2 = B
        fc = 1/(2*OSR_N) - B/2
        fc1 = -fc
        fc2 = fc
    elif scenario =='dual-band_asymmetric':
        symmetric = False
        B1 = np.ones(4)*1/24
        B2 = np.ones(4)*1/48
        OSR_L= 1/(B1+B2)
        OSR_N = np.array([1.25, 2, 4, 8])
        fc1 = -0.3*np.ones(4)
        delta_f = 1/OSR_N - (B1+B2)/2
        fc2 = fc1 + delta_f


    n  = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]) # sample size
    data_dict = {}
    data_dict['scenario'] = scenario
    data_dict['n'] = n
    data_dict['num_MC'] = num_MC
    data_dict['symmetric'] = symmetric
    data_dict['OSR_L'] = OSR_L
    data_dict['OSR_N'] = OSR_N
    data_dict['B1'] = B1
    data_dict['B2'] = B2
    data_dict['fc1'] = fc1
    data_dict['fc2'] = fc2


    # Monte Carlo simulation
    num_filters = len(OSR_L)
    PAPR_MC = np.zeros((num_filters, len(n), num_MC))
    t_start = time.time()
    filter_order = 64
    print(scenario+'\n')
    for i in range(num_filters):
        print(f'fc1 = {fc1[i]:.3f}, B1 = {B1[i]:.3f}, fc2 = {fc2[i]:.3f}, B2 = {B2[i]:.3f} cycles per sample\n')
        if symmetric:  # real-valued filter coeffs
            sos1 = design_iir_filter(filter_order, symmetric, fc1[i], B1[i])
            sos2 = None
            data_dict[f'filt{i}_sos1'] = sos1
        else: # complex-valued filter coeffs
            sos1 = design_iir_filter(filter_order, symmetric, fc1[i], B1[i])
            data_dict[f'filt{i}_sos1_real'] = sos1.real
            data_dict[f'filt{i}_sos1_imag'] = sos1.imag
            if B2[i] == 0:
                sos2 = None
            else:
                sos2 = design_iir_filter(filter_order, symmetric, fc2[i], B2[i])
                data_dict[f'filt{i}_sos2_real'] = sos2.real
                data_dict[f'filt{i}_sos2_imag'] = sos2.imag

        for k in range(len(n)):
            print(f'computing Monte Carlo results for n={int(n[k])} \n')
            PAPR_MC[i,k,:] = compute_PAPR_filtered_WGN_MC_parallel(int(n[k]), num_MC, filt_sos1=sos1, filt_sos2=sos2, batch_size=1000)
    data_dict['PAPR_MC'] = PAPR_MC
    t_end = time.time()
    print(f'Monte Carlo processing time = {(t_end - t_start)/60:.1f} minutes \n')
    with open(fname, 'w') as f:
        json.dump(data_dict, f, cls=NumpyArrayEncoder)



#%%
if __name__ == "__main__":
    main()