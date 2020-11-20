import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython import embed
from pathlib import Path
sns.set()

###########################################################
# Plot model paths and uncertainties
# ###########################################################
# Parameters:

base_folder = Path(os.environ['FENICS_ICE_BASE_DIR']) / "example_cases"

run_folders = ['ismipc_rc_1e4','ismipc_rc_1e5','ismipc_rc_1e6']


# Output Directory
outdir = Path('/home/dgoldber/www/public_html/fenics_ice')
outdir.mkdir(parents=True, exist_ok=True)
#########################

upperlim = 0;

f, axarr = plt.subplots(1,2, sharex=True)

for i, rf in enumerate(run_folders):

    #run_dir = base_folder / rf
    #result_dir = run_dir / "output"
    #Qfile = 'Qval_ts.p'
    #sigmafile = 'sigma.p'
    #sigmapriorfile = 'sigma_prior.p'

    run_dir = base_folder / rf
    result_dir = run_dir / "output"
    Qfile = "_".join((rf, 'Qval_ts.p'))
    sigmafile = "_".join((rf, 'sigma.p'))
    sigmapriorfile = "_".join((rf, 'sigma_prior.p'))

    # pd = pickle.load(open(os.path.join(base_folder, rf, Qfile), 'rb'))
    pd = pickle.load((result_dir / Qfile).open('rb'))
    dQ_vals = pd[0]
    dQ_t = pd[1]

    pd = pickle.load((result_dir / sigmafile).open('rb'))
    sigma_vals = pd[0]
    sigma_t = pd[1]

    pd = pickle.load((result_dir / sigmapriorfile).open('rb'))
    sigma_prior_vals = pd[0]

    sigma_interp = np.interp(dQ_t, sigma_t, sigma_vals)
    sigma_prior_interp = np.interp(dQ_t, sigma_t, sigma_prior_vals)
    
    QoIs_MC = np.load(result_dir/"sampling_results.npy")
    QoIs_MC = QoIs_MC[:,QoIs_MC[-1,:]>0]
    print( str(np.shape(QoIs_MC)) + ' values OK')
    sigma_mc_vals = np.std(QoIs_MC,1)
    sigma_mc_interp = np.interp(dQ_t, sigma_t, sigma_mc_vals)


    x = dQ_t
    y = dQ_vals - dQ_vals[0]
    s = sigma_interp
    sp = sigma_prior_interp
    sm = sigma_mc_interp
    upperlim = max(upperlim,np.max(sp))
    upperlim = max(upperlim,np.max(s))
    upperlim = max(upperlim,np.max(sm))


    axarr[0].plot(x, y, 'k' if i == 0 else 'k:')
#    axarr[0].fill_between(x, y-sp, y+sp, facecolor='b')
    axarr[0].fill_between(x, y-s, y+s, facecolor='g' if i == 0 else 'b', alpha = .5 if i==0 else .6)
#    axarr[0].fill_between(x, y-sm, y+sm, facecolor='r' if i == 0 else 'y', alpha = .5 if i==0 else .6)

    axarr[1].semilogy(x, sp, 'b' if i == 0 else 'g')
    axarr[1].semilogy(x, s, 'b:' if i == 0 else 'g:')
    if (i==0):
     clr = 'blue'
    else:
     clr = 'green'
    axarr[1].semilogy(sigma_t, sigma_mc_vals,color=clr,marker='+',linestyle='none',markersize=6)

axarr[0].set_xlabel('Time (yrs)')
axarr[1].set_xlabel('Time (yrs)')
axarr[0].set_ylabel(r'$Q$ $(m^4)$')
axarr[1].set_ylabel(r'$\sigma$ $(m^4)$')
axarr[1].set_ylim([1e6,upperlim*1.1])
axarr[0].text(0.05, 0.95, 'a', transform=axarr[0].transAxes,
fontsize=13, fontweight='bold', va='top')

axarr[1].text(0.05, 0.95, 'b', transform=axarr[1].transAxes,
fontsize=13, fontweight='bold', va='top')
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join(outdir,'run_paths_sample.pdf'), bbox_inches="tight")
