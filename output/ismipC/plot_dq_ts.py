 # Plots the sensitivity of the QOI to the parameter of interest. This sensitivity
# is determined by run_forward.py, either at only the final timestep, or at a given
# interval. 

# If only one sensitivity was calculated, then setting n_sens = 0 returns
# the sensitivity at the end of the run. If more than one sensitivity was calculated, then n_sens = 0
# returns the value at timestep = 0 (so it is constant), and n_sens = (the value passed to --num_sens - 1) 
# is the last timestep
 
# Parameters:
n_sens = 4
base_folder = '/mnt/c/Users/ckozi/Documents/Python/fenics/fenics_ice/output/ismipC'
run_folders = [
    f'{base_folder}/ismipC_inv4_perbc_20x20_gnhep_prior/run_forward',]

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


fig = plt.figure()

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

    dQ = Function(Qp)

    x    = mesh.coordinates()[:,0]
    y    = mesh.coordinates()[:,1]
    t    = mesh.cells()

    hdf5data = HDF5File(MPI.comm_world, os.path.join(rf, 'dQ_ts.h5'), 'r')
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

plt.savefig('dq_ts.pdf')
plt.show()
