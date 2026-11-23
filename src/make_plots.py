# -*- coding: utf-8 -*-
"""
Script for plots in the paper
"Extreme Value Theory for Peak Power in Discrete-Time Complex Gaussian Processes"

author: Adam Wunderlich
last update: 11/19/2026
"""

import json
from json import JSONEncoder
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.stats import gaussian_kde

import PAPR_distribution_utils as pdu


class NumpyArrayEncoder(JSONEncoder):  # custom json encoder for numpy arrays
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


#%%
def main():

    #%%

    num_MC =  int(1e5) # number of MC trials per condition
    scenario = 'dual-band_asymmetric' # 'single-band_symmetric', 'dual-band_symmetric', 'dual-band_asymmetric'
    fname = scenario  + f'-{num_MC:.0e}'.replace('+','') + '.json'
    
    colors = ["#004949","#009292","#ff6db6","#006ddb","#b66dff","#6db6ff",
              "#924900","#db6d00","#24ff24","#ffff6d"]

    if os.path.isfile(fname): # load prior Monte Carlo results
        with open(fname) as f:
            data_dict = json.load(f)
            try:
                scenario = data_dict['scenario'] = scenario
                symmetric = data_dict['symmetric']
                n = np.asarray(data_dict['n'])
                num_MC = data_dict['num_MC'] 
                OSR_L = np.asarray(data_dict['OSR_L'])
                OSR_N = np.asarray(data_dict['OSR_N'])
                B1 = np.asarray(data_dict['B1'])
                B2 = np.asarray(data_dict['B2'])
                fc1 = np.asarray(data_dict['fc1'])
                fc2 = np.asarray(data_dict['fc2'])
                PAPR_MC = np.asarray(data_dict['PAPR_MC'])    
                W1_CT = np.asarray(data_dict['W1_CT']) 
                W1_DT = np.asarray(data_dict['W1_DT'])
                W1_iid = np.asarray(data_dict['W1_iid'])
            except Exception:
                pass
   
    num_filters = len(OSR_L)
    
    # plots
    fig1, ax = plt.subplots(2, 2, figsize = (8,6), sharex=True, sharey=True)
    ax_flat = ax.flatten()
    filt_inds = range(num_filters)
    for ax_ind, filt_ind in enumerate(filt_inds):
        ax_flat[ax_ind].plot(n[:], W1_iid[filt_ind,:], color = colors[3], linestyle = '-', marker = '^', fillstyle = 'none', markeredgewidth=2.5, linewidth = 3, markersize = 12, label = 'i.i.d.') 
        ax_flat[ax_ind].plot(n, (W1_CT[filt_ind,:]), color = colors[6], linestyle = '-', marker = 'v', fillstyle = 'none', markeredgewidth=2.5, linewidth = 3, markersize = 12, label = 'CT')   
        ax_flat[ax_ind].plot(n, (W1_DT[filt_ind,:]), color = colors[1], linestyle = '-', marker = 's', fillstyle = 'none', markeredgewidth=2.5, linewidth = 3, markersize = 12, label = 'DT')   
        ax_flat[ax_ind].grid()
        ax_flat[ax_ind].set_xscale('log')
        ax_flat[ax_ind].tick_params(axis='both', which='major', labelsize=14)
        ax_flat[ax_ind].tick_params(axis='both', which='minor', labelsize=14)
        ax_flat[ax_ind].set_title(rf'$\mathrm{{OSR}}_N = ${OSR_N[ax_ind]:g}, $\mathrm{{OSR}}_L = ${OSR_L[ax_ind]:g}', fontsize=16)
    ax_flat[0].set_ylabel('$W_1$ (dB)', fontsize = 16)
    ax_flat[1].legend(loc='best', ncol=1, fontsize = 12)
    ax_flat[0].set_ylim(-0.1,1.65)
    ax_flat[0].set_yticks([0,.25, .5, .75, 1, 1.25, 1.5])
    ax_flat[2].set_ylabel('$W_1$ (dB)', fontsize = 16)
    ax_flat[2].set_xlabel('Number of Samples', fontsize = 16)
    ax_flat[3].set_xlabel('Number of Samples', fontsize = 16)
    fig1.tight_layout()
    fig1.savefig(scenario+'.png', dpi=400)
    
    # plot PAPR pdfs 
    fig2, ax = plt.subplots(1, 2, figsize = (8,4), sharey = True)
    for ax_ind in range(2):
        if ax_ind ==0:
            i = 0
        else:
            i = 3
        k = 2 # index for n
        papr_grid_dB = np.linspace(6.5, 12, 500)
        PAPR_sample_dB = 10*np.log10(PAPR_MC[i,k,:]) # CDFs are on dB scale
        kernel = gaussian_kde(PAPR_sample_dB)
        Delta, kappa = pdu.spectral_parameters(B1[i], fc1[i], B2[i], fc2[i])
        # numerically differentiate CDFs to get PDFs
        PAPR_iid_pdf_dB = pdu.PAPR_PDF_iid(papr_grid_dB, n[k])
        PAPR_CT_pdf_dB = pdu.PAPR_PDF_CT(papr_grid_dB, Delta, n[k])
        PAPR_DT_pdf_dB = np.diff(pdu.PAPR_CDF_DT(papr_grid_dB, kappa, n[k]))/ np.diff(papr_grid_dB)
        papr_grid_dB_adjusted = (papr_grid_dB[:-1] + papr_grid_dB[1:])/2 # adjust grid for differentiation
        ax[ax_ind].plot(papr_grid_dB, PAPR_iid_pdf_dB, color = colors[3], marker = '^', fillstyle = 'none', markeredgewidth=2.5, markersize = 12, markevery = 0.1, linewidth = 3, label = 'i.i.d.')
        ax[ax_ind].plot(papr_grid_dB, PAPR_CT_pdf_dB, color = colors[6], marker = 'v', fillstyle = 'none', markeredgewidth=2.5, markersize = 12, markevery = 0.1, linewidth = 3, label = 'CT')
        ax[ax_ind].plot(papr_grid_dB_adjusted, PAPR_DT_pdf_dB, color = colors[1], marker = 's', fillstyle = 'none', markeredgewidth=2.5, markersize = 12, markevery = 0.1, linewidth = 3, label = 'DT')
        ax[ax_ind].plot(papr_grid_dB, kernel(papr_grid_dB), color = 'k', marker = 'o', markevery = 0.1, markersize = 12, linewidth = 3, label='empirical')
        ax[ax_ind].grid()
        ax[ax_ind].set_xlim([papr_grid_dB[0], papr_grid_dB[-1]])
        #ax[ax_ind].set_xticks([7,8,9,10,11,12])
        #ax[ax_ind].set_ylim([-0.035, 0.95])
        ax[ax_ind].set_xlabel('PAPR (dB)', fontsize = 16)       
        ax[ax_ind].tick_params(axis='both', which='major', labelsize=16)
        ax[ax_ind].tick_params(axis='both', which='minor', labelsize=16)
        ax[ax_ind].set_title(rf'$\mathrm{{OSR}}_N = ${OSR_N[i]:g}, $\mathrm{{OSR}}_L = ${OSR_L[i]:g}', fontsize=16)
        
    ax[0].set_ylabel('Probability Density', fontsize = 16)
    ax[0].legend(loc='best', fontsize = 11)
    fig2.tight_layout()
    fig2.savefig(scenario+'_pdfs'+'.png', dpi=400)

    # plot filter frequency responses
    fig3, ax = plt.subplots(2, 2, figsize = (8,6), sharex=True, sharey=True)
    ax_flat = ax.flatten()

    for i in range(num_filters):
        if symmetric:      
            sos = np.asarray(data_dict[f'filt{i}_sos1'])
            w, H = signal.freqz_sos(sos, whole=True, fs=1.0)
            w = np.fft.fftshift(w) 
            w[w >= 0.5] -= 1.0 
            H = np.fft.fftshift(H)
        else: 
            sos1_real = np.asarray(data_dict[f'filt{i}_sos1_real'])
            sos1_imag = np.asarray(data_dict[f'filt{i}_sos1_imag']) 
            sos1 = sos1_real + 1j*sos1_imag
            sos2_real = np.asarray(data_dict[f'filt{i}_sos2_real']) 
            sos2_imag = np.asarray(data_dict[f'filt{i}_sos2_imag']) 
            sos2 = sos2_real + 1j*sos2_imag
            w, H1 = signal.freqz_sos(sos1, whole=True, fs=1.0)
            w, H2 = signal.freqz_sos(sos2, whole=True, fs=1.0)
            w = np.fft.fftshift(w) 
            w[w >= 0.5] -= 1.0 
            H1 = np.fft.fftshift(H1)
            H2 = np.fft.fftshift(H2)
            H = H1 + H2
        ax_flat[i].plot(w,20*np.log10(np.abs(H)), color = 'k', linewidth=3)
        ax_flat[i].grid(True)
        ax_flat[i].set_xlim(-0.5,0.5)
        ax_flat[i].set_ylim(-80,3)
        ax_flat[i].set_yticks([-80,-60,-40,-20,0])
        if i > 1:
            ax_flat[i].set_xlabel('Digital Frequency (cycles/sample)', fontsize = 14)
        if i == 0 or i == 2:
            ax_flat[i].set_ylabel('Filter Gain (dB)', fontsize = 16)
        ax_flat[i].tick_params(axis='both', which='major', labelsize=14)
        ax_flat[i].tick_params(axis='both', which='minor', labelsize=14)
        ax_flat[i].set_title(rf'$\mathrm{{OSR}}_N = ${OSR_N[i]:g}, $\mathrm{{OSR}}_L = ${OSR_L[i]:g}', fontsize=16)
        fig3.tight_layout()
        fig3.savefig(scenario + '_filter_responses.png', dpi=400)

   
    if not symmetric:
        # teaser plots
        i = 1 # index for filter number
        k = 2 # index for n
        fig4, ax4 = plt.subplots(1, 1, figsize = (6,4.5), sharey = True)
        sos1_real = np.asarray(data_dict[f'filt{i}_sos1_real'])
        sos1_imag = np.asarray(data_dict[f'filt{i}_sos1_imag']) 
        sos1 = sos1_real + 1j*sos1_imag
        sos2_real = np.asarray(data_dict[f'filt{i}_sos2_real']) 
        sos2_imag = np.asarray(data_dict[f'filt{i}_sos2_imag']) 
        sos2 = sos2_real + 1j*sos2_imag
        w, H1 = signal.freqz_sos(sos1, whole=True, fs=1.0)
        w, H2 = signal.freqz_sos(sos2, whole=True, fs=1.0)
        w = np.fft.fftshift(w) 
        w[w >= 0.5] -= 1.0 
        H1 = np.fft.fftshift(H1)
        H2 = np.fft.fftshift(H2)
        H = H1 + H2
        ax4.plot(w,20*np.log10(np.abs(H)), color = 'k', linewidth=3)
        ax4.grid(True)
        ax4.set_xlim(-0.5,0.5)
        ax4.set_ylim(-80,3)
        ax4.set_yticks([-80,-60,-40,-20,0])
        ax4.set_xlabel('Digital Frequency (cycles/sample)', fontsize = 18)
        ax4.set_ylabel('Filter Gain (dB)', fontsize = 18)
        ax4.tick_params(axis='both', which='major', labelsize=16)
        ax4.tick_params(axis='both', which='minor', labelsize=16)
        fig4.tight_layout()
        fig4.savefig(scenario + '_teaser_filter_response.png', dpi=400)
        
        fig5, ax5 = plt.subplots(1, 1, figsize = (6,4.5))
        papr_grid_dB = np.linspace(7, 12, 500)
        PAPR_sample_dB = 10*np.log10(PAPR_MC[i,k,:]) # CDFs are on dB scale
        kernel = gaussian_kde(PAPR_sample_dB)
        
        Delta, kappa = pdu.spectral_parameters(B1[i], fc1[i], B2[i], fc2[i])
       
        PAPR_iid_pdf_dB = pdu.PAPR_PDF_iid(papr_grid_dB, n[k])
        PAPR_CT_pdf_dB = pdu.PAPR_PDF_CT(papr_grid_dB, Delta, n[k])
        PAPR_DT_pdf_dB = np.diff(pdu.PAPR_CDF_DT(papr_grid_dB, kappa, n[k]))/ np.diff(papr_grid_dB)
        papr_grid_dB_adjusted = (papr_grid_dB[:-1] + papr_grid_dB[1:])/2 # adjust grid for differentiation    
        ax5.plot(papr_grid_dB, PAPR_iid_pdf_dB, color = colors[3], marker = '^', fillstyle = 'none', markeredgewidth=2.5, markersize = 12, markevery = 0.1, linewidth = 3, label = 'i.i.d.')
        ax5.plot(papr_grid_dB, PAPR_CT_pdf_dB, color = colors[6], marker = 'v', fillstyle = 'none', markeredgewidth=2.5, markersize = 12, markevery = 0.1, linewidth = 3, label = 'CT')
        ax5.plot(papr_grid_dB_adjusted, PAPR_DT_pdf_dB, color = colors[1], marker = 's', fillstyle = 'none', markeredgewidth=2.5, markersize = 12, markevery = 0.1, linewidth = 3, label = 'DT')
        ax5.plot(papr_grid_dB, kernel(papr_grid_dB), color = 'k', marker = 'o', markevery = 0.1, markersize = 12, linewidth = 3, label='empirical')
        ax5.grid()
        ax5.set_xlim([papr_grid_dB[0], papr_grid_dB[-1]])
        #ax5.set_ylim([-0.03, 0.85])
        ax5.set_xlabel('PAPR (dB)', fontsize = 18)       
        ax5.tick_params(axis='both', which='major', labelsize=16)
        ax5.tick_params(axis='both', which='minor', labelsize=16)
        ax5.set_ylabel('Probability Density', fontsize = 18)
        ax5.legend(loc='upper left', fontsize = 14)
        fig5.tight_layout()
        fig5.savefig(scenario+'_teaser_pdf'+'.png', dpi=400)

        # example non-monotonic ACF magnitude
        Rzz_length = 10
        Rzz = np.ones(Rzz_length, dtype=complex)
        def sinc(x):
            if x==0:
                return 1
            else:
                return np.sin(np.pi*x)/(np.pi*x)
            
        for k in np.arange(Rzz_length):
            if k > 0:
                Rzz[k] = (B1[i]*np.exp(1j*2*np.pi*fc1[i]*k)*sinc(B1[i]*k) + B2[i]*np.exp(1j*2*np.pi*fc2[i]*k)*sinc(B2[i]*k))/(B1[i]+B2[i])
  
       
        fig, ax = plt.subplots(1, 1, figsize = (6,4))
        ax.stem(np.arange(Rzz_length),np.abs(Rzz), basefmt="gray")
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
        ax.set_xlabel('k', fontsize = 20)
        ax.set_ylabel('$|r_{ZZ}[k]|$', fontsize = 20)     
        fig.tight_layout()
        fig.savefig('kappa_hat'+'.png', dpi=400)
#%%
if __name__ == "__main__":
    main()