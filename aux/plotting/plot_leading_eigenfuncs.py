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

from fenics_ice.backend import Function, FunctionSpace, HDF5File, MPI, Mesh

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from pathlib import Path

from fenics_ice import model, config
from fenics_ice import mesh as fice_mesh


###########################################################
# Plot four eigenfunctions. Defaults to leading four
###########################################################
# Parameters:

#Offset from first eigenvector (0 results in leading four)
e_offset = 0    

base_folder = Path(os.environ['FENICS_ICE_BASE_DIR']) / "example_cases"

# Simulation Directories: A list of one or more directories
run_folders = [
    'ismipc_rc_1e6',
    'ismipc_rc_1e4',
    ]

#Figure size in inches (width, height). Passed to Pyplot figure();
figsize = (18, 6)

# Output Directory
outdir = base_folder / "plots"
outdir.mkdir(parents=True, exist_ok=True)

#########################

cmap='Blues'
cmap_div='RdBu'
numlev = 40
tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


fig = plt.figure(figsize=figsize)
fig.tight_layout()

for i, rf in enumerate(run_folders):

    run_dir = base_folder / rf
    mesh = Mesh(str(run_dir / "input" / "ismip_mesh.xml"))

    param_file = str((run_dir / rf).with_suffix(".toml"))
    params = config.ConfigParser(param_file, top_dir=run_dir)

    Q = FunctionSpace(mesh,'Lagrange',1)
    Qh = FunctionSpace(mesh,'Lagrange',3)
    M = FunctionSpace(mesh,'DG',0)

    if not params.mesh.periodic_bc:
        Qp = Q
    else:
        Qp = fice_mesh.get_periodic_space(params, mesh, dim=1)

    eigenfunc = Function(Qp)

    x    = mesh.coordinates()[:,0]
    y    = mesh.coordinates()[:,1]
    t    = mesh.cells()

    for j in range(4):
        k = j + e_offset
        vr_file = str(next((run_dir / 'output').glob("*vr.h5")))
        hdf5data = HDF5File(MPI.comm_world, vr_file, 'r')
        hdf5data.read(eigenfunc, f'v/vector_{k}')

        sind = j+1+i*4
        ax  = fig.add_subplot(2,4,sind)
        ax.text(0.05, 0.95, labels[sind-1], transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

        ax.set_aspect('equal')
        v   = np.abs(eigenfunc.compute_vertex_values(mesh))
        minv = np.min(v)
        maxv = np.max(v)
        levels = np.linspace(minv,maxv,numlev)
        ticks = np.linspace(minv,maxv,3)
        ax.tick_params(**tick_options)

        c = ax.tricontourf(x, y, t, v, levels = levels, cmap=plt.get_cmap(cmap_div))
        cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="vertical")

fig = plt.gcf()
plt.show()
fig.savefig(os.path.join(outdir,'leading_eigenvectors.pdf'), bbox_inches="tight")
