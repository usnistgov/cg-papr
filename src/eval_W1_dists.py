# -*- coding: utf-8 -*-
"""
Script to calculate W1 distances for evaluations in the paper
"Extreme Value Theory for Peak Power in Discrete-Time Complex Gaussian Processes"

author: Adam Wunderlich
last update: 11/18/2026
"""

import numpy as np
import time
import os
import json
import PAPR_distribution_utils as pdu
from json import JSONEncoder
from functools import partial


class NumpyArrayEncoder(JSONEncoder):  # custom json encoder for numpy arrays
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

#%%
def main():

    #%%

    num_MC =  int(1e5) # number of MC trials per condition
    scenario = 'dual-band_asymmetric' #'single-band_symmetric', 'dual-band_symmetric', 'dual-band_asymmetric'
    fname = scenario  + f'-{num_MC:.0e}'.replace('+','') + '.json'

    if os.path.isfile(fname): # load prior Monte Carlo results
        with open(fname) as f:
            data_dict = json.load(f)
            try:
                scenario = data_dict['scenario'] = scenario
                n = np.asarray(data_dict['n'])
                num_MC = data_dict['num_MC'] 
                OSR_L = np.asarray(data_dict['OSR_L'])
                OSR_N = np.asarray(data_dict['OSR_N'])
                B1 = np.asarray(data_dict['B1'])
                B2 = np.asarray(data_dict['B2'])
                fc1 = np.asarray(data_dict['fc1'])
                fc2 = np.asarray(data_dict['fc2'])
                PAPR_MC = np.asarray(data_dict['PAPR_MC'])    
            except Exception:
                pass
            
    print('computing performance metrics')
    num_filters = len(OSR_L)
    W1_iid = np.zeros((num_filters,len(n)))
    W1_iid_CI = np.zeros((num_filters,len(n),2))    
    W1_CT = np.zeros((num_filters,len(n)))
    W1_CT_CI = np.zeros((num_filters,len(n),2))
    W1_DT = np.zeros((num_filters,len(n)))
    W1_DT_CI = np.zeros((num_filters,len(n),2))
    t_start = time.time()
    for i in range(num_filters):       
        print(f'computing perf results for case {i+1} of {num_filters} \n')
        Delta, kappa = pdu.spectral_parameters(B1[i], fc1[i], B2[i], fc2[i])
        for k in range(len(n)):
            print(f'computing perf results for n={int(n[k])} \n')
            PAPR_sample_dB = 10*np.log10(PAPR_MC[i,k,:]) # CDFs are on dB scale
            CDF_CT = partial(pdu.PAPR_CDF_CT, Delta=Delta, n=n[k])   
            CDF_iid = partial(pdu.PAPR_CDF_iid, n=n[k])
            CDF_DT = partial(pdu.PAPR_CDF_DT, kappa=kappa, n=n[k])
            W1_CT[i,k], W1_CT_CI[i,k,:] = pdu.W1_distance(PAPR_sample_dB, CDF_CT, estimate_CI = False)  
            W1_iid[i,k], W1_iid_CI[i,k,:] = pdu.W1_distance(PAPR_sample_dB, CDF_iid, estimate_CI = False)  
            W1_DT[i,k], W1_DT_CI[i,k,:] = pdu.W1_distance(PAPR_sample_dB, CDF_DT, estimate_CI = False)  
  
    t_end = time.time()
    print(f'performance result processing time = {(t_end - t_start)/60:.1f} minutes \n')
    data_dict['W1_CT'] = W1_CT
    data_dict['W1_CT_CI'] = W1_CT_CI
    data_dict['W1_DT'] = W1_DT
    data_dict['W1_DT_CI'] = W1_DT_CI
    data_dict['W1_iid'] = W1_iid
    data_dict['W1_iid_CI'] = W1_iid_CI

    with open(fname, 'w') as f:
        json.dump(data_dict, f, cls=NumpyArrayEncoder)
        
#%%
if __name__ == "__main__":
    main()