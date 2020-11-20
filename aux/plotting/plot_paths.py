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
import seaborn as sns
import os
from pathlib import Path
sns.set()

###########################################################
# Plot model paths and uncertainties
# ###########################################################
# Parameters:

base_folder = Path(os.environ['FENICS_ICE_BASE_DIR']) / "example_cases"

run_folders = ['ismipc_Q4','ismipc_rc_1e4']

# Output Directory
outdir = base_folder / "plots"
outdir.mkdir(parents=True, exist_ok=True)
#########################


f, axarr = plt.subplots(1,2, sharex=True)

for i, rf in enumerate(run_folders):

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

    x = dQ_t
    y = dQ_vals - dQ_vals[0]
    s = 2*sigma_interp
    sp = 2*sigma_prior_interp

    axarr[0].plot(x, y, 'k' if i == 0 else 'k:')
#    axarr[0].fill_between(x, y-sp, y+sp, facecolor='b')
    axarr[0].fill_between(x, y-s, y+s)

    axarr[1].semilogy(x, sp, 'b' if i == 0 else 'b:')
    axarr[1].semilogy(x, s, 'g' if i == 0 else 'g:')

axarr[0].set_xlabel('Time (yrs)')
axarr[1].set_xlabel('Time (yrs)')
axarr[0].set_ylabel(r'$Q$ $(m^4)$')
axarr[1].set_ylabel(r'$\sigma$ $(m^4)$')
#axarr[1].set_ylim([1,10**10])
axarr[0].text(0.05, 0.95, 'a', transform=axarr[0].transAxes,
fontsize=13, fontweight='bold', va='top')

axarr[1].text(0.05, 0.95, 'b', transform=axarr[1].transAxes,
fontsize=13, fontweight='bold', va='top')
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join(outdir,'run_paths.pdf'), bbox_inches="tight")
