import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from IPython import embed
import sys

###########################################################
# Plot the eigenvalue decay for a run(s)
###########################################################
# Parameters:

base_folder = Path(os.environ['FENICS_ICE_BASE_DIR']) / "example_cases"

if (len(sys.argv)==1):
    run_folders = ['ismipc_rc_1e6',
                   'ismipc_30x30',
                   'ismipc_40x40'
                   ]
    labels = ('Low Res', 'Mid Res', 'High Res')
    outdir = base_folder / "plots"
elif(len(sys.argv)==2):
    run_folders = ['ismipc_rc_1e6',
                   'ismipc_30x30',
                   'ismipc_40x40'
                   ]
    labels = ('Low Res', 'Mid Res', 'High Res')
    outdir = Path(sys.argv[1])
else:
    outdir = Path(sys.argv[1])
    figname = Path(sys.argv[2])
    run_folders=sys.argv[3:-1:2]
    labels = tuple(sys.argv[4::2])
    
#    for i in range(len(run_folders)):
#        run_folders[i] = run_folders[i]+'/output'

print(labels)

# Output Directory
outdir.mkdir(parents=True, exist_ok=True)
#########################

plt.figure()
print(outdir)
for i, rf in enumerate(run_folders):
    print(rf)

    lamfile = "_".join((rf,'eigvals.p'))

    pd = pickle.load(open(os.path.join(base_folder, rf, 'output', lamfile), 'rb'))
    lam = pd[0]
    lpos = np.argwhere(lam > 0)
    lneg = np.argwhere(lam < 0)
    lind = np.arange(0,len(lam))
    plt.semilogy(lind[lpos], 1./(1+lam[lpos]), '.', alpha = 0.5, mew=0, label =labels[i])
    plt.semilogy(lind[lneg], 1./(1+np.abs(lam[lneg])), '.k', alpha = 0.12, mew=0,)


plt.legend()
plt.xlabel('Eigenvalue index')
plt.ylabel('Uncertainty reduction')
plt.savefig(os.path.join(outdir,figname), bbox_inches="tight")
