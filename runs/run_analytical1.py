import sys
from fenics import *
sys.path.insert(0,'../code/')
import model
import solver
import matplotlib.pyplot as plt
from IPython import embed

#Load Data

dd = '../input/analytical1/'
data_mesh = Mesh(''.join([dd,'analytical1_mesh.xml']))
Q = FunctionSpace(data_mesh, 'Lagrange', 1)
bed = Function(Q,''.join([dd,'analytical1_mesh_bed.xml']))
surf = Function(Q,''.join([dd,'analytical1_mesh_surf.xml']))
bmelt = Function(Q,''.join([dd,'analytical1_mesh_bmelt.xml']))
B2 = Function(Q,''.join([dd,'analytical1_mesh_B2.xml']))
alpha = ln(B2)

#Generate model mesh
nx = 481
ny = 481
L = 240e3
mesh = RectangleMesh(Point(0,0), Point(L, L), nx, ny)


#Initialize Model
param = {'eq_def' : 'action',
        'outdir' :'./output_analytical1/'}
mdl = model.model(mesh,param)
mdl.init_surf(surf)
mdl.init_bed(bed)
mdl.gen_thick()
mdl.init_bmelt(bmelt)
mdl.init_alpha(alpha)

mdl.gen_ice_mask()
mdl.gen_domain()

#Solve
slvr = solver.ssa_solver(mdl)
slvr.def_mom_eq()
slvr.solve_mom_eq()

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
