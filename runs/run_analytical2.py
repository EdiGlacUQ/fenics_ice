import sys
from fenics import *
sys.path.insert(0,'../code/')
import model
import solver
import matplotlib.pyplot as plt
from IPython import embed
import numpy as np

#Load Data

dd = '../input/analytical2/'
data_mesh = Mesh(''.join([dd,'analytical2_mesh.xml']))
Q = FunctionSpace(data_mesh, 'Lagrange', 1)
bed = Function(Q,''.join([dd,'analytical2_mesh_bed.xml']))
surf = Function(Q,''.join([dd,'analytical2_mesh_surf.xml']))
bmelt = Function(Q,''.join([dd,'analytical2_mesh_bmelt.xml']))
bdrag = Function(Q,''.join([dd,'analytical2_mesh_bdrag.xml']))

#Number of cells in grid
nx = 50;
ny = 400;

#Fenics mesh
Lx = 50e3
Ly = 400e3

mesh = RectangleMesh(Point(0,0), Point(Lx, Ly), nx, ny)


#Initialize Model
#eq_def=1 SSA from Action Principle (Default)
#eq_def=2 SSA directly in weak form
output_dir='./output_analytical2/'
mdl = model.model(mesh,outdir=output_dir,eq_def=2)
mdl.init_surf(surf)
mdl.init_bed(bed)
mdl.init_thick()
mdl.init_bmelt(bmelt)
mdl.init_bdrag(bdrag)
mdl.default_solver_params()

mdl.gen_ice_mask()
mdl.gen_domain()

#Solve
slvr = solver.ssa_solver(mdl)
slvr.def_mom_eq()
slvr.solve_mom_eq()


#Plot
x = np.linspace(0,Lx,nx+1)
points = [(x_,Ly/2) for x_ in x]
mod_line = np.array([slvr.U(point)[0] for point in points])
ex_line = x * (mdl.rhoi*mdl.g*mdl.delta*1000.0/4.0)**3 * mdl.A
plt.figure()
plt.plot(ex_line)
plt.plot(mod_line)
plt.savefig('tmp.png')

vtkfile = File(''.join([mdl.outdir,'U.pvd']))
U = project(mdl.U,mdl.V)
vtkfile << U

vtkfile = File(''.join([mdl.outdir,'bed.pvd']))
vtkfile << mdl.bed

vtkfile = File(''.join([mdl.outdir,'surf.pvd']))
vtkfile << mdl.surf

vtkfile = File(''.join([mdl.outdir,'thick.pvd']))
vtkfile << mdl.thick

vtkfile = File(''.join([mdl.outdir,'mask.pvd']))
vtkfile << mdl.mask
