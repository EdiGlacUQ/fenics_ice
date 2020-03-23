import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from fenics import *
import model

###########################################################
# Plot the result of an inversion. This shows:
# 1. The inverted value of B2; It is explicitly assumed that B2 = alpha**2
# 2. The the standard deviation of alpha.
# 3. The resulting velocities obtained by solving the momentum equations using the inverted value of B2.
# 4. The difference between observed and modelled velocities
###########################################################
# Parameters:

# Simulation Directory
dd = '/mnt/c/Users/ckozi/Documents/Python/fenics/fenics_ice/output/ismipC/uq_rc_1e4'

# Output Directory
outdir = os.path.join(dd, 'plots')

###########################################################

if not os.path.isdir(outdir):
    print('Outdir does not exist. Creating...')
    os.mkdir(outdir)

cmap='Blues'
cmap_div='RdBu'
numlev = 20
tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

mesh = Mesh(os.path.join(dd,'mesh.xml'))
param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )

Q = FunctionSpace(mesh,'Lagrange',1)
Qh = FunctionSpace(mesh,'Lagrange',3)
M = FunctionSpace(mesh,'DG',0)

if not param['periodic_bc']:
   Qp = Q
   V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
else:
   Qp = FunctionSpace(mesh,'Lagrange',1,constrained_domain=model.PeriodicBoundary(param['periodic_bc']))
   V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2,constrained_domain=model.PeriodicBoundary(param['periodic_bc']))

U = Function(V,os.path.join(dd,'U.xml'))
alpha = Function(Qp,os.path.join(dd,'alpha.xml'))
uv_obs = Function(M, os.path.join(dd,'uv_obs.xml'))
alpha_sigma = Function(Qp, os.path.join(dd,'run_forward/alpha_sigma.xml'))
# B2 = Function(M, os.path.join(dd,'B2.xml'))

u, v = U.split()
uv = project(sqrt(u*u + v*v), Q)
uv_diff = project(uv_obs - uv, Q)
B2 = project(alpha*alpha, M)


x    = mesh.coordinates()[:,0]
y    = mesh.coordinates()[:,1]
t    = mesh.cells()


fig = plt.figure(figsize=(10,5))


ax  = fig.add_subplot(151)
ax.set_aspect('equal')
v    = B2.compute_vertex_values(mesh)
minv = np.min(v)
maxv = np.max(v)
levels = np.linspace(minv,maxv,numlev)
ticks = np.linspace(minv,maxv,3)
ax.tick_params(**tick_options)
ax.text(0.05, 0.95, 'a', transform=ax.transAxes,
    fontsize=13, fontweight='bold', va='top')
c = ax.tricontourf(x, y, t, v, levels = levels, cmap=plt.get_cmap(cmap))
cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal")
cbar.ax.set_xlabel(r'${B^2}$ (Pa $m^{-1}$ yr)')

ax  = fig.add_subplot(152)
ax.set_aspect('equal')
v    = alpha_sigma.compute_vertex_values(mesh)
minv = np.min(v)
maxv = np.max(v)
levels = np.linspace(minv,maxv,numlev)
ticks = np.linspace(minv,maxv,3)
ax.tick_params(**tick_options)
ax.text(0.05, 0.95, 'b', transform=ax.transAxes,
    fontsize=13, fontweight='bold', va='top')
c = ax.tricontourf(x, y, v, levels = levels, cmap=plt.get_cmap(cmap))
cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal", format=ticker.FormatStrFormatter('%1.1e'))
cbar.ax.xaxis.set_major_locator(ticker.LinearLocator(3))

cbar.ax.set_xlabel(r'$\sigma$ ($Pa^{0.5}$ $m^{-0.5}$ $yr^{0.5}$)')

ax  = fig.add_subplot(153)
ax.set_aspect('equal')
v   = uv.compute_vertex_values(mesh)
levels = np.linspace(10,30,numlev)
ticks = np.linspace(10,30,3)
ax.tick_params(**tick_options)
ax.text(0.05, 0.95, 'c', transform=ax.transAxes,
    fontsize=13, fontweight='bold', va='top')
c = ax.tricontourf(x, y, t, v, levels = levels, cmap=plt.get_cmap(cmap))
cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal")
cbar.ax.set_xlabel(r'$U$ (m $yr^{-1}$)')


ax  = fig.add_subplot(154)
ax.set_aspect('equal')
v   = uv_obs.compute_vertex_values(mesh)
levels = np.linspace(10,30,numlev)
ticks = np.linspace(10,30,3)
ax.tick_params(**tick_options)
ax.text(0.05, 0.95, 'd', transform=ax.transAxes,
    fontsize=13, fontweight='bold', va='top')
c = ax.tricontourf(x, y, t, v, levels = levels, cmap=plt.get_cmap(cmap))
cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal")
cbar.ax.set_xlabel(r'$U_{obs}$ (m $yr^{-1}$)')

ax  = fig.add_subplot(155)
ax.set_aspect('equal')
v   = uv_diff.compute_vertex_values(mesh)
max_diff = np.rint(np.max(np.abs(v)))
levels = np.linspace(-max_diff,max_diff,numlev)
ticks = np.linspace(-max_diff,max_diff,3)
ax.tick_params(**tick_options)
ax.text(0.05, 0.95, 'e', transform=ax.transAxes,
    fontsize=13, fontweight='bold', va='top')
c = ax.tricontourf(x, y, t, v, levels = levels, cmap=plt.get_cmap(cmap_div))
cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal")
cbar.ax.set_xlabel(r'$U-U_{obs}$ (m $yr^{-1}$)')
plt.tight_layout(2.0)
plt.savefig(os.path.join(outdir, 'inv_results.pdf'))
plt.show()
