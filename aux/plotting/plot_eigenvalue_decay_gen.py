import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython import embed
from pathlib import Path
import sys

###########################################################
# Plot the eigenvalue decay for a run(s)
###########################################################
# Parameters:

base_folder = Path(os.environ['FENICS_ICE_BASE_DIR']) / "example_cases"

if (len(sys.argv)==1):
    run_folders = ['ismipc_rc_1e6/output',
                   'ismipc_30x30/output',
                   'ismipc_40x40/output'
                   ]
    labels = ('Low Res', 'Mid Res', 'High Res')
else:
    run_folders=sys.argv[1:]
    labels = tuple(sys.argv[1:])
    for i in range(len(run_folders)):
        run_folders[i] = run_folders[i]+'/output'
    
print(run_folders)    

#Legend values for simulations


# Output Directory
outdir = base_folder / "plots"
outdir.mkdir(parents=True, exist_ok=True)
#########################

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
