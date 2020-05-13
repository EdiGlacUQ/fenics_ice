import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from fenics import *
from fenics_ice import model


###########################################################
# Plot four eigenfunctions. Defaults to leading four
###########################################################
# Parameters:

#Offset from first eigenvector (0 results in leading four)
e_offset = 0    

base_folder = os.path.join(os.environ['FENICS_ICE_BASE_DIR'], 'output/ismipC')

# Simulation Directories: A list of one or more directories
run_folders = [
    'uq_rc_1e6/run_forward',
    'uq_rc_1e4/run_forward',
    ]

#Figure size in inches (width, height). Passed to Pyplot figure();
figsize = (18, 6)

# Output Directory
outdir = os.path.join(base_folder, 'plots')

#########################

if not os.path.isdir(outdir):
    print('Outdir does not exist. Creating...')
    os.mkdir(outdir)


cmap='Blues'
cmap_div='RdBu'
numlev = 40
tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


fig = plt.figure(figsize=figsize)
fig.tight_layout()

for i, rf in enumerate(run_folders):
    mesh = Mesh(os.path.join(base_folder, rf,'mesh.xml'))
    param = pickle.load( open( os.path.join(base_folder, rf,'param.p'), "rb" ) )

    Q = FunctionSpace(mesh,'Lagrange',1)
    Qh = FunctionSpace(mesh,'Lagrange',3)
    M = FunctionSpace(mesh,'DG',0)

    if not param['periodic_bc']:
       Qp = Q
    else:
       Qp = FunctionSpace(mesh,'Lagrange',1,constrained_domain=model.PeriodicBoundary(param['periodic_bc']))

    eigenfunc = Function(Qp)

    x    = mesh.coordinates()[:,0]
    y    = mesh.coordinates()[:,1]
    t    = mesh.cells()

    for j in range(4):
        k = j + e_offset
        hdf5data = HDF5File(MPI.comm_world, os.path.join(base_folder, rf, 'vr.h5'), 'r')
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
