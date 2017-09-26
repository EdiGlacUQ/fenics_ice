import sys
sys.path.insert(0,'../code/')
from fenics import *
import model
import solver
import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
from IPython import embed
#Load Data

dd = '../input/grnld/'
data_mesh = Mesh(''.join([dd,'grnld_mesh.xml']))
Q = FunctionSpace(data_mesh, 'Lagrange', 1)
bed = Function(Q,''.join([dd,'grnld_mesh_bed.xml']))
surf = Function(Q,''.join([dd,'grnld_mesh_surf.xml']))
bmelt = Function(Q,''.join([dd,'grnld_mesh_bmelt.xml']))
B2 = Function(Q,''.join([dd,'grnld_mesh_B2.xml']))
alpha = ln(B2)

#Generate model mesh
nx = 150
ny = 150

mesh = RectangleMesh(Point(0,0), Point(150e3, 150e3), nx, ny)



#Initialize Model
param = {'eq_def' : 'action',
        'outdir' :'./output_grnld_inv/'}
mdl = model.model(mesh,param)
mdl.init_surf(surf)
mdl.init_bed(bed)
mdl.init_thick()
mdl.init_bmelt(bmelt)
mdl.init_alpha(alpha)

mdl.gen_ice_mask()
mdl.gen_domain()

#Solve
slvr = solver.ssa_solver(mdl)
slvr.def_mom_eq()
slvr.solve_mom_eq()

#Inversions
set_log_level(40)
u,v = split(slvr.U)
mdl.init_vel_obs(u,v)
mdl.init_alpha(Constant(ln(1500)))
slvr = solver.ssa_solver(mdl)
slvr.inversion()


#Plot
alpha_inv = project(exp(slvr.alpha_inv),mdl.Q)
F_vals = [x for x in slvr.F_vals if x > 0]

fu.plot_variable(alpha_inv, 'alpha_inverted', mdl.param['outdir'])
fu.plot_inv_conv(F_vals, 'convergence', mdl.param['outdir'])
