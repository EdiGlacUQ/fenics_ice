# Plot model paths and uncertainties 

run_folders = [
    './ismipC_inv4_perbc_20x20_gnhep_prior/run_forward',
    './ismipC_inv6_perbc_20x20_gnhep_prior/run_forward',]

#########################

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


run_folders = [
    './ismipC_inv4_perbc_20x20_gnhep_prior/run_forward',
    './ismipC_inv6_perbc_20x20_gnhep_prior/run_forward',]

run_folders = [
'./uq_rc_1e4/run_forward',
'./uq_rc_1e4/run_forward',]

f, axarr = plt.subplots(1,2, sharex=True)

for i, rf in enumerate(run_folders):

    Qfile = 'Qval_ts.p'
    sigmafile = 'sigma.p'
    sigmapriorfile = 'sigma_prior.p'

    pd = pickle.load(open(os.path.join(rf, Qfile), 'rb'))
    dQ_vals = pd[0]
    dQ_t = pd[1]

    pd = pickle.load(open(os.path.join(rf, sigmafile), 'rb'))
    sigma_vals = pd[0]
    sigma_t = pd[1]

    pd = pickle.load(open(os.path.join(rf, sigmapriorfile), 'rb'))
    sigma_prior_vals = pd[0]

    sigma_interp = np.interp(dQ_t, sigma_t, sigma_vals)
    sigma_prior_interp = np.interp(dQ_t, sigma_t, sigma_prior_vals)

    x = dQ_t
    y = dQ_vals - dQ_vals[0]
    s = 2*sigma_interp
    sp = 2*sigma_prior_interp

    axarr[0].plot(x, y, 'k' if i == 0 else 'k:')
    axarr[0].fill_between(x, y-sp, y+sp)
    axarr[0].fill_between(x, y-s, y+s)

    axarr[1].semilogy(x, sp, 'b' if i == 0 else 'b:')
    axarr[1].semilogy(x, s, 'g' if i == 0 else 'g:')

axarr[0].set_xlabel('Time (yrs)')
axarr[1].set_xlabel('Time (yrs)')
axarr[0].set_ylabel(r'$Q$ $(m^4)$')
axarr[1].set_ylabel(r'$\sigma$ $(m^4)$')

axarr[0].text(0.05, 0.95, 'a', transform=axarr[0].transAxes,
fontsize=13, fontweight='bold', va='top')

axarr[1].text(0.05, 0.95, 'b', transform=axarr[1].transAxes,
fontsize=13, fontweight='bold', va='top')
plt.show()
plt.savefig('run_paths.pdf')
