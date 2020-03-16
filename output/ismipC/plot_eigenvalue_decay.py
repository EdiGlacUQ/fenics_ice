# Plot the eigenvalue decay for a run(s)

base_folder = '/mnt/c/Users/ckozi/Documents/Python/fenics/fenics_ice/output/ismipC'
run_folders = ['ismipC_inv6_perbc_20x20_gnhep_prior/run_forward',
    'ismipC_inv6_perbc_30x30_gnhep_prior/run_forward',
    'ismipC_inv6_perbc_40x40_gnhep_prior/run_forward'
    ]
#########################




import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

plt.figure()
labels = []
for i, rf in enumerate(run_folders):

    lamfile = 'slepceig_all.p'


    pd = pickle.load(open(os.path.join(rf, lamfile), 'rb'))
    lam = pd[0]
    lpos = np.argwhere(lam > 0)
    lneg = np.argwhere(lam < 0)
    lind = np.arange(0,len(lam))
    plt.semilogy(lind[lpos], lam[lpos], '.')
    #plt.semilogy(lind[lneg], np.abs(lam[lneg]), '.k')

plt.legend({'Low Res', 'Mid Res', 'High Res'})
plt.xlabel('Eigenvalue')
plt.ylabel('Magnitude')
plt.savefig('grid_convergence.pdf')
plt.show()
