# For fenics_ice copyright information see ACKNOWLEDGEMENTS in the fenics_ice
# root directory

# This file is part of fenics_ice.
#
# fenics_ice is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# fenics_ice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

###########################################################
# Plot the eigenvalue decay for a run(s)
###########################################################
# Parameters:

base_folder = Path(os.environ['FENICS_ICE_BASE_DIR']) / "example_cases"

run_folders = ['ismipc_rc_1e4'
    ]

#Legend values for simulations
labels = ('Low Res', 'Mid Res', 'High Res')

# Output Directory
outdir = base_folder / "plots"
outdir.mkdir(parents=True, exist_ok=True)
#########################

plt.figure()
for i, rf in enumerate(run_folders):
    print(rf)

    lamfile = "_".join((rf,'eigvals.p'))

    pd = pickle.load(open(os.path.join(base_folder, rf, 'output', lamfile), 'rb'))
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
