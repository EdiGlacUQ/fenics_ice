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

#Store key python variables in here
savefile = 'smithinv_' + datetime.datetime.now().strftime("%m%d%H%M")

set_log_level(20)

#Load Data
dd = '../input/smith_500m_input/'
data_mesh = Mesh(''.join([dd,'smith450m_mesh.xml']))
Q = FunctionSpace(data_mesh, 'DG', 0)
bed = Function(Q,''.join([dd,'smith450m_mesh_bed.xml']), name = "bed")
thick = Function(Q,''.join([dd,'smith450m_mesh_thick.xml']), name = "thick")
mask = Function(Q,''.join([dd,'smith450m_mesh_mask.xml']), name = "mask")
u_obs = Function(Q,''.join([dd,'smith450m_mesh_u_obs.xml']), name = "u_obs")
v_obs = Function(Q,''.join([dd,'smith450m_mesh_v_obs.xml']), name = "v_obs")
u_std = Function(Q,''.join([dd,'smith450m_mesh_u_std.xml']), name = "u_std")
v_std = Function(Q,''.join([dd,'smith450m_mesh_v_std.xml']), name = "v_std")
mask_vel = Function(Q,''.join([dd,'smith450m_mesh_mask_vel.xml']), name = "mask_vel")
B_mod = Function(Q,''.join([dd,'smith450m_mesh_mask_B_mod.xml']), name = "B_mod")

#Generate model mesh
gf = 'grid_data.npz'
npzfile = np.load(''.join([dd,'grid_data.npz']))
nx = int(npzfile['nx'])
ny = int(npzfile['ny'])
xlim = npzfile['xlim']
ylim = npzfile['ylim']

mesh = RectangleMesh(Point(xlim[0],ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny, 'crossed')

#Initialize Model
param = {'eq_def' : 'weak',
        'solver': 'petsc',
        'outdir' :'./output_smith_inv/',
        'rc_inv': [1, 1e-4, 5e6, 40.0], #alpha only
        #'rc_inv': [1e-5, 1e-4, 100.0, 40.0], #alpha + beta
        'inv_options': {'disp': True, 'maxiter': 5}
        }


mdl = model.model(mesh,mask, param)
mdl.init_bed(bed)
mdl.init_thick(thick)
mdl.gen_surf()
mdl.init_mask(mask)
mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
mdl.init_lat_dirichletbc()
mdl.init_bmelt(Constant(0.0))
mdl.gen_alpha()
#mdl.init_alpha(Constant(ln(20000))) #Initialize using uniform alpha
mdl.init_beta(ln(B_mod))            #Comment to use uniform Bglen
mdl.label_domain()


slvr = solver.ssa_solver(mdl)

# alpha0 = slvr.alpha.copy(deepcopy=True)
# cc = Control(alpha0)
#
# slvr.taylor_ver(alpha0,annotate_flag=True)
# dJ = compute_gradient(Functional(slvr.J), cc, forget = False)
# ddJ = hessian(Functional(slvr.J), cc)

cc = Control(slvr.alpha)

slvr.taylor_ver2(slvr.alpha)
dJ = compute_gradient(Functional(slvr.J), cc, forget = False)
ddJ = hessian(Functional(slvr.J), cc)

minconv = taylor_test(slvr.taylor_ver, cc, assemble(slvr.J), dJ, HJm = ddJ, seed = 1e-1, size = 7)
