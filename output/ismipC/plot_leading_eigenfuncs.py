# Plot four eigenfunctions. Defaults to leading four

# Parameters:
e_offset = 0    #Offset from first eigenvalue (0 results in leading four)

run_folders = [
    './ismipC_inv4_perbc_20x20_gnhep_prior/run_forward',
    './ismipC_inv6_perbc_20x20_gnhep_prior/run_forward',]

#########################


import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from fenics import *
import model


cmap='Blues'
cmap_div='RdBu'
numlev = 40
tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


fig = plt.figure()
fig.tight_layout()

for i, rf in enumerate(run_folders):
    mesh = Mesh(os.path.join(rf,'mesh.xml'))
    param = pickle.load( open( os.path.join(rf,'param.p'), "rb" ) )

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
        hdf5data = HDF5File(MPI.comm_world, os.path.join(rf, 'vr.h5'), 'r')
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

plt.show()
plt.savefig('leading_eigenvalues.pdf')