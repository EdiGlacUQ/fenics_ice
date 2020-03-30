import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

###########################################################
# Plot the eigenvalue decay for a run(s)
###########################################################
# Parameters:

base_folder = os.path.join(os.environ['FENICS_ICE_BASE_DIR'], 'output/ismipC')

run_folders = ['uq_rc_1e6/run_forward',
    'uq_30x30/run_forward',
    'uq_40x40/run_forward'
    ]

#Legend values for simulations
labels = ('Low Res', 'Mid Res', 'High Res')

# Output Directory
outdir = os.path.join(base_folder, 'plots')
#########################

if not os.path.isdir(outdir):
    print('Outdir does not exist. Creating...')
    os.mkdir(outdir)

plt.figure()
for i, rf in enumerate(run_folders):
    print(rf)

    lamfile = 'slepc_eig_all.p'

    pd = pickle.load(open(os.path.join(base_folder, rf, lamfile), 'rb'))
    lam = pd[0]
    lpos = np.argwhere(lam > 0)
    lneg = np.argwhere(lam < 0)
    lind = np.arange(0,len(lam))
    plt.semilogy(lind[lpos], lam[lpos], '.', alpha = 0.5, mew=0, label =labels[i])
    plt.semilogy(lind[lneg], np.abs(lam[lneg]), '.k', alpha = 0.12, mew=0,)


plt.legend()
plt.xlabel('Eigenvalue')
plt.ylabel('Magnitude')
plt.savefig(os.path.join(outdir,'grid_convergence.pdf'), bbox_inches="tight")
plt.show()
