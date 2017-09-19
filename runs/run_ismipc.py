import sys
from fenics import *
sys.path.insert(0,'../code/')
import model
import solver
import matplotlib.pyplot as plt
from IPython import embed

#Load Data

dd = '../input/ismipC/'
data_mesh = Mesh(''.join([dd,'ismipC_mesh.xml']))
Q = FunctionSpace(data_mesh, 'Lagrange', 1)
bed = Function(Q,''.join([dd,'ismipC_mesh_bed.xml']))
surf = Function(Q,''.join([dd,'ismipC_mesh_surf.xml']))
bmelt = Function(Q,''.join([dd,'ismipC_mesh_bmelt.xml']))
bdrag = Function(Q,''.join([dd,'ismipC_mesh_bdrag.xml']))

#Generate model mesh
nx = 120
ny = 120
L = 120e3
mesh = RectangleMesh(Point(0,0), Point(L, L), nx, ny)


#Initialize Model
#eq_def=1 SSA from Action Principle (Default)
#eq_def=2 SSA directly in weak form
mdl = model.model(mesh,eq_def=1)
mdl.init_surf(surf)
mdl.init_bed(bed)
mdl.init_thick()
mdl.init_bmelt(bmelt)
mdl.init_bdrag(bdrag)
mdl.default_solver_params()

mdl.gen_ice_mask()
mdl.gen_boundaries()

#Solve
slvr = solver.ssa_solver(mdl)
slvr.def_mom_eq()
slvr.solve_mom_eq()

vtkfile = File('U.pvd')
U = project(mdl.U,mdl.V)
vtkfile << U

vtkfile = File('bed.pvd')
vtkfile << mdl.bed

vtkfile = File('surf.pvd')
vtkfile << mdl.surf

vtkfile = File('thick.pvd')
vtkfile << mdl.thick

vtkfile = File('mask.pvd')
vtkfile << mdl.mask

vtkfile = File('bdrag.pvd')
vtkfile << mdl.bdrag
