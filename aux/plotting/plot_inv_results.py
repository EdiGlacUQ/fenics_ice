import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
from fenics import *
from fenics_ice import model, config
from fenics_ice import mesh as fice_mesh
from pathlib import Path
sns.set()

###########################################################
# Plot the result of an inversion. This shows:
# 1. The inverted value of B2; It is explicitly assumed that B2 = alpha**2
# 2. The the standard deviation of alpha.
# 3. The resulting velocities obtained by solving the momentum equations using the inverted value of B2.
# 4. The observed velocities
# 5. The difference between observed and modelled velocities
###########################################################
# Parameters:

# Simulation Directory
run_name = 'ismipc_rc_1e6'
dd = Path(os.environ['FENICS_ICE_BASE_DIR']) / 'example_cases' / run_name

# Output Directory
outdir = dd / "plots"
outdir.mkdir(exist_ok=True)

results_dir = dd / "output"
###########################################################

cmap='Blues'
cmap_div='RdBu'
numlev = 20
tick_options = {'axis':'both','which':'both','bottom':False,
    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}

mesh = Mesh(str(dd / "input" / "ismip_mesh.xml"))
#param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )

param_file = str((dd/run_name).with_suffix(".toml"))
params = config.ConfigParser(param_file, top_dir=dd)

Q = FunctionSpace(mesh,'Lagrange',1)
Qh = FunctionSpace(mesh,'Lagrange',3)
M = FunctionSpace(mesh,'DG',0)

if not params.mesh.periodic_bc:
   Qp = Q
   V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
else:
    Qp = fice_mesh.get_periodic_space(params, mesh, dim=1)
    V =  fice_mesh.get_periodic_space(params, mesh, dim=2)

U_file = str(next(results_dir.glob("*U.xml")))
alpha_file = str(next(results_dir.glob("*alpha.xml")))
uv_obs_file = str(next(results_dir.glob("*uv_obs.xml")))
alpha_sigma_file = str(next(results_dir.glob("*alpha_sigma.xml")))


U = Function(V, U_file)
alpha = Function(Qp, alpha_file)
uv_obs = Function(M, uv_obs_file)
alpha_sigma = Function(Qp, alpha_sigma_file)
# B2 = Function(M, os.path.join(dd,'B2.xml'))

u, v = U.split()
uv = project(sqrt(u*u + v*v), Q)
uv_diff = project(uv_obs - uv, Q)
B2 = project(alpha*alpha, M)


x    = mesh.coordinates()[:,0]
y    = mesh.coordinates()[:,1]
t    = mesh.cells()


fig = plt.figure(figsize=(10,5))


ax  = fig.add_subplot(231)
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

ax  = fig.add_subplot(232)
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

ax  = fig.add_subplot(233)
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


ax  = fig.add_subplot(234)
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

ax  = fig.add_subplot(235)
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
