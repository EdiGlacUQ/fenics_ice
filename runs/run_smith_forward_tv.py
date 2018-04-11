import sys
sys.path.insert(0,'../code/')
from fenics import *
from dolfin_adjoint import *

import model
import solver
import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
import time
import datetime
import pickle
from IPython import embed

set_log_level(20)

#Load Data
dd = './output_smith_inv/'
param = pickle.load( open( ''.join([dd,'param.p']), "rb" ) )
param['outdir'] = './output_smith_forward/'

data_mesh = Mesh(''.join([dd,'data_mesh.xml']))
M_dm = FunctionSpace(data_mesh,'DG',0)
data_mask = Function(M_dm,''.join([dd,'data_mask.xml']))

mdl_mesh = Mesh(''.join([dd,'mesh.xml']))

V = VectorFunctionSpace(mdl_mesh,'Lagrange',1,dim=2)
Q = FunctionSpace(mdl_mesh,'Lagrange',1)
M = FunctionSpace(mdl_mesh,'DG',0)

U = Function(V,''.join([dd,'U.xml']))
alpha = Function(Q,''.join([dd,'alpha.xml']))
beta = Function(Q,''.join([dd,'beta.xml']))
bed = Function(Q,''.join([dd,'bed.xml']))
surf = Function(Q,''.join([dd,'surf.xml']))
thick = Function(M,''.join([dd,'thick.xml']))
mask = Function(M,''.join([dd,'mask.xml']))
mask_vel = Function(M,''.join([dd,'mask_vel.xml']))
u_obs = Function(M,''.join([dd,'u_obs.xml']))
v_obs = Function(M,''.join([dd,'v_obs.xml']))
u_std = Function(M,''.join([dd,'u_std.xml']))
v_std = Function(M,''.join([dd,'v_std.xml']))
uv_obs = Function(M,''.join([dd,'uv_obs.xml']))
Bglen = Function(M,''.join([dd,'Bglen.xml']))
B2 = Function(Q,''.join([dd,'B2.xml']))


param['run_length'] =  0.2
param['n_steps'] = 2

mdl = model.model(data_mesh,data_mask, param)
mdl.init_bed(bed)
mdl.init_thick(thick)
mdl.gen_surf()
mdl.init_mask(mask)
mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
mdl.init_lat_dirichletbc()
mdl.init_bmelt(Constant(0.0))
mdl.init_alpha(alpha)
mdl.init_beta(beta)
mdl.label_domain()


#Solve
H_init = mdl.H_s.copy(deepcopy=True)

slvr = solver.ssa_solver(mdl)
slvr.save_H_init(H_init)

alpha0 = slvr.alpha.copy(deepcopy=True)
slvr.taylor_ver_vaf(alpha0,adjoint_flag=1)
cc = Control(slvr.alpha)
dJ = compute_gradient(Functional(slvr.J_vaf), cc, forget = False)

minconv = taylor_test(slvr.taylor_ver_vaf, cc, assemble(slvr.J_vaf), dJ, seed = 1e0, size = 7)
