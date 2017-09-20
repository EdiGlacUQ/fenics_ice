import sys
from fenics import *
sys.path.insert(0,'../code/')
import model
import solver
import matplotlib.pyplot as plt
from IPython import embed

#Load Data

dd = '../input/grnld/'
data_mesh = Mesh(''.join([dd,'grnld_mesh.xml']))
Q = FunctionSpace(data_mesh, 'Lagrange', 1)
bed = Function(Q,''.join([dd,'grnld_mesh_bed.xml']))
surf = Function(Q,''.join([dd,'grnld_mesh_surf.xml']))
bmelt = Function(Q,''.join([dd,'grnld_mesh_bmelt.xml']))
bdrag = Function(Q,''.join([dd,'grnld_mesh_bdrag.xml']))

#Generate model mesh
nx = 150
ny = 150
mesh = RectangleMesh(Point(0,0), Point(150e3, 150e3), nx, ny)


#Initialize Model
#eq_def=1 SSA from Action Principle (Default)
#eq_def=2 SSA directly in weak form
output_dir='./output2/'
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
