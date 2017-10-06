from fenics import *
import model
import solver
import fenics_util as fu
import matplotlib.pyplot as plt


#Load Data
data_mesh = Mesh('gldbg2013_mesh.xml')
Q = FunctionSpace(data_mesh, 'Lagrange', 1)
bed = Function(Q,'gldbg2013_mesh_bed.xml')
surf = Function(Q,'gldbg2013_mesh_surf.xml')
bmelt = Function(Q,'gldbg2013_mesh_bmelt.xml')
B2 = Function(Q,'gldbg2013_mesh_B2.xml')
alpha = ln(B2)

#Generate model mesh
nx = 150
ny = 150
mesh = RectangleMesh(Point(0,0), Point(150e3, 150e3), nx, ny)




#Initialize Model

param = {'eq_def' : 'action',
        'outdir' :'./output_gldbg2013/'}
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

#vtkfile = File('thick.pvd')
#vtkfile << B
