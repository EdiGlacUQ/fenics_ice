import sys
from fenics import *
from dolfin_adjoint import *
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
thick = Function(M,''.join([dd,'ismipC_mesh_height.xml']))
bmelt = Function(M,''.join([dd,'ismipC_mesh_bmelt.xml']))
mask = Function(M,''.join([dd,'ismipC_mesh_mask.xml']))
B2 = Function(M,''.join([dd,'ismipC_mesh_B2.xml']))
alpha = sqrt(B2)

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
        'rhoi': 910.0 }
mdl = model.model(mesh,mask, param)
mdl.init_bed(bed)
mdl.init_thick(thick)
mdl.gen_surf()
mdl.init_mask(mask)
mdl.init_bmelt(bmelt)
mdl.init_alpha(alpha)
mdl.label_domain()

slvr = solver.ssa_solver(mdl)

alpha0 = slvr.alpha.copy(deepcopy=True)
cc = Control(alpha0)

slvr.taylor_ver(alpha0,annotate_flag=True)
dJ = compute_gradient(Functional(slvr.J), cc, forget = False)
ddJ = hessian(Functional(slvr.J), cc)

minconv = taylor_test(slvr.taylor_ver, cc, assemble(slvr.J), dJ,HJm = ddJ, seed = 1e-2, size = 4)
