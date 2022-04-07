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

from fenics_ice.backend import Function, FunctionSpace, HDF5File, Mesh, MPI

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from fenics_ice import model, config
from fenics_ice import mesh as fice_mesh
from pathlib import Path

###########################################################
# Plots the sensitivity of the QOI to the parameter of interest. This sensitivity
# is determined by run_forward.py, either at only the final timestep, or at a given
# interval. 

# If only one sensitivity was calculated, then setting n_sens = 0 returns
# the sensitivity at the end of the run. If more than one sensitivity was calculated, then n_sens = 0
# returns the value at timestep = 0 (so it is constant), and n_sens = (the value passed to --num_sens - 1) 
# is the last timestep
# ###########################################################
# Parameters:

#The sensitivity to look at
n_sens = 4

base_folder = Path(os.environ['FENICS_ICE_BASE_DIR']) / "example_cases"
run_folders = ['ismipc_rc_1e6','ismipc_rc_1e4']

# Output Directory
plot_outdir = base_folder / "plots"
plot_outdir.mkdir(parents=True, exist_ok=True)


#########################


cmap='Blues'
cmap_div='RdBu'
numlev = 40
tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}


fig = plt.figure()

for i, rf in enumerate(run_folders):


    run_dir = base_folder / rf

    param_file = str((run_dir/rf).with_suffix(".toml"))
    params = config.ConfigParser(param_file, top_dir=run_dir)

    outdir = run_dir / params.io.output_dir
    indir = run_dir / params.io.input_dir

    mesh = Mesh(str(indir/'ismip_mesh.xml'))

#    param = pickle.load( open( os.path.join(base_folder, rf,'param.p'), "rb" ) )

    Q = FunctionSpace(mesh,'Lagrange',1)
    Qh = FunctionSpace(mesh,'Lagrange',3)
    M = FunctionSpace(mesh,'DG',0)

    if not params.mesh.periodic_bc:
       Qp = Q
    else:
       Qp = fice_mesh.get_periodic_space(params, mesh, dim=1)

    dQ = Function(Qp)

    x    = mesh.coordinates()[:,0]
    y    = mesh.coordinates()[:,1]
    t    = mesh.cells()

    hdffile = str(next(outdir.glob("*dQ_ts.h5")))
    hdf5data = HDF5File(MPI.comm_world, hdffile, 'r')
    hdf5data.read(dQ, f'dQ/vector_{n_sens}')




    ax  = fig.add_subplot(1,2,i+1)
    ax.set_aspect('equal')
    v   = dQ.compute_vertex_values(mesh)
    minv = np.min(v)
    maxv = np.max(v)
    mmv = np.max([np.abs(minv),np.abs(maxv)])
    levels = np.linspace(-mmv,mmv,numlev)
    ticks = np.linspace(-mmv,mmv,3)
    ax.tick_params(**tick_options)
    ax.text(0.05, 0.95, 'ab'[i], transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='top')
    c = ax.tricontourf(x, y, t, v, levels = levels, cmap=plt.get_cmap(cmap_div))
    cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal", format="%.1E")
    cbar.ax.set_xlabel(r'$\frac{dQ}{d\alpha}$')

plt.savefig(os.path.join(plot_outdir,'dq_ts.pdf'), bbox_inches="tight")
plt.show()
