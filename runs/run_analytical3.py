import sys
from fenics import *
sys.path.insert(0,'../code/')
import model
import solver
import matplotlib.pyplot as plt
from IPython import embed
import numpy as np

#Load Data

dd = '../input/analytical3/'
data_mesh = Mesh(''.join([dd,'analytical3_mesh.xml']))
M = FunctionSpace(data_mesh, 'DG', 0)
bed = Function(M,''.join([dd,'analytical3_mesh_bed.xml']))
height = Function(M,''.join([dd,'analytical3_mesh_height.xml']))
bmelt = Function(M,''.join([dd,'analytical3_mesh_bmelt.xml']))
mask = Function(M,''.join([dd,'analytical3_mesh_mask.xml']))
mask = project(conditional(gt(mask,1.0-1e-2), 1.0, 0.0), M) #Binary
B2 = Function(M,''.join([dd,'analytical3_mesh_B2.xml']))
alpha = ln(B2)

#Number of cells in grid
nx = 50;
ny = 400;

#Fenics mesh
Lx = 50e3
Ly = 400e3

mesh = RectangleMesh(Point(0,0), Point(Lx, Ly), nx, ny)

#Initialize Model
param = {'eq_def' : 'weak',
        'outdir' :'./output_analytical3/'}
mdl = model.model(mesh,param)
mdl.init_bed(bed)
mdl.init_thick(height)
mdl.init_mask(mask)
mdl.gen_vel_mask()
mdl.init_bmelt(bmelt)
mdl.init_alpha(alpha)

#mdl.gen_ice_mask()
mdl.gen_domain()

#Solve
slvr = solver.ssa_solver(mdl)
slvr.def_mom_eq()
slvr.solve_mom_eq()

#Plot
outdir = mdl.param['outdir']

rhoi = mdl.param['rhoi']
rhow = mdl.param['rhow']
g = mdl.param['g']
delta = 1.0 - rhoi/rhow
A = mdl.param['A']
h = 1000.0

x = np.linspace(0,Lx,nx+1)
points = [(x_,Ly/2) for x_ in x]
mod_line = np.array([slvr.U(point)[0] for point in points])
ex_line = x * (rhoi*g*delta*h/4.0)**3 * A
plt.figure()
plt.plot(ex_line, color='k')
plt.plot(mod_line, 'b.')
plt.savefig(outdir + 'lineplot.png')


vtkfile = File(''.join([outdir,'U.pvd']))
U = project(mdl.U,mdl.V)
vtkfile << U

vtkfile = File(''.join([outdir,'bed.pvd']))
vtkfile << mdl.bed

vtkfile = File(''.join([outdir,'thick.pvd']))
vtkfile << mdl.thick

vtkfile = File(''.join([outdir,'mask.pvd']))
vtkfile << mdl.mask
