import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

###########################################################
# Plot the eigenvalue decay for a run(s)
###########################################################
# Parameters:

base_folder = '/mnt/c/Users/ckozi/Documents/Python/fenics/fenics_ice/output/ismipC'

run_folders = ['ismipC_inv6_perbc_20x20_gnhep_prior/run_forward',
    'ismipC_inv6_perbc_30x30_gnhep_prior/run_forward',
    'ismipC_inv6_perbc_40x40_gnhep_prior/run_forward'
    ]

#Legend values for simulations
labels = ('Low Res', 'Mid Res', 'High Res')

# Output Directory
outdir = '/mnt/c/Users/ckozi/Documents/Python/fenics/fenics_ice/output/ismipC'
#########################


plt.figure()
for i, rf in enumerate(run_folders):
    print(rf)

    lamfile = 'slepceig_all.p'


    pd = pickle.load(open(os.path.join(base_folder, rf, lamfile), 'rb'))
    lam = pd[0]
    lpos = np.argwhere(lam > 0)
    lneg = np.argwhere(lam < 0)
    lind = np.arange(0,len(lam))
    plt.semilogy(lind[lpos], lam[lpos], '.', alpha = 0.5, mew=0, label =labels[i])
    #plt.semilogy(lind[lneg], np.abs(lam[lneg]), '.k')

plt.legend()
plt.xlabel('Eigenvalue')
plt.ylabel('Magnitude')
plt.savefig(os.path.join(outdir,'grid_convergence.pdf'))
plt.show()
