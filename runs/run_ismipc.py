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
M = FunctionSpace(data_mesh, 'DG', 0)
bed = Function(M,''.join([dd,'ismipC_mesh_bed.xml']))
height = Function(M,''.join([dd,'ismipC_mesh_height.xml']))
bmelt = Function(M,''.join([dd,'ismipC_mesh_bmelt.xml']))
mask = Function(M,''.join([dd,'ismipC_mesh_mask.xml']))
B2 = Function(M,''.join([dd,'ismipC_mesh_B2.xml']))
alpha = ln(B2)

#Generate model mesh
nx = 120
ny = 120
L = 120e3
mesh = RectangleMesh(Point(0,0), Point(L, L), nx, ny)


#Initialize Model
param = {'eq_def' : 'weak',
        'solver': 'petsc',
        'outdir' :'./output_ismipc/',
        'A': 10**(-16),
        'rhoi': 910 }
mdl = model.model(mesh,mask, param)
mdl.init_bed(bed)
mdl.init_thick(height)
mdl.init_mask(mask)
mdl.init_bmelt(bmelt)
mdl.init_alpha(alpha)

mdl.label_domain()

#Solve
slvr = solver.ssa_solver(mdl)
slvr.def_mom_eq()
slvr.solve_mom_eq()

#Plot
outdir = mdl.param['outdir']

vtkfile = File(''.join([outdir,'U.pvd']))
U = project(slvr.U,slvr.V)
vtkfile << U

vtkfile = File(''.join([outdir,'bed.pvd']))
vtkfile << mdl.bed

vtkfile = File(''.join([outdir,'thick.pvd']))
vtkfile << mdl.thick

vtkfile = File(''.join([outdir,'mask.pvd']))
vtkfile << mdl.mask

vtkfile = File(''.join([outdir,'B2.pvd']))
B2 = project(exp(mdl.alpha), M)
vtkfile << B2

embed()
