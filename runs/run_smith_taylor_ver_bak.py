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


def forward(alpha = None):
    mdl = model.model(mesh,mask, param)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
    mdl.init_lat_dirichletbc()
    mdl.init_bmelt(Constant(0.0))
    if alpha is None:
        mdl.gen_alpha()
        #mdl.init_alpha(Constant(ln(6000))) #Initialize using uniform alpha
        alpha = mdl.alpha
    else:
        mdl.init_alpha(alpha)
    mdl.init_beta(ln(B_mod))            #Comment to use uniform Bglen

    mdl.label_domain()

    #Solve
    slvr = solver.ssa_solver(mdl)
    slvr.def_mom_eq()
    slvr.solve_mom_eq()

    return alpha, mdl, slvr

alpha0, mdl, slvr = forward()

#embed()
cc = Control(alpha0)
u,v = split(slvr.U)

#J = (mdl.u_std**(-2)*(u-mdl.u_obs)**2 + mdl.v_std**(-2)*(v-mdl.v_obs)**2)*slvr.dObs
J = inner(slvr.U,slvr.U)*slvr.dIce
grad_alpha = grad(alpha0)
grad_alpha_ = project(grad_alpha, mdl.V)
lap_alpha = div(grad_alpha_)
lambda_a = mdl.param['rc_inv'][0]
delta_a = mdl.param['rc_inv'][2]
reg_a = lambda_a * alpha0 - delta_a*lap_alpha
reg_a_bndry = delta_a*inner(grad_alpha,mdl.nm)
J_reg_alpha = inner(reg_a,reg_a)*slvr.dIce + inner(reg_a_bndry,reg_a_bndry)*slvr.ds
parameters["adjoint"]["stop_annotating"] = True
#adj_html("forward.html", "forward")

#J += J_reg_alpha

J_val = assemble(J)
dJ = compute_gradient(Functional(J), cc, forget = False)
ddJ = hessian(Functional(J),cc)
def J_test(alpha):
    _, mdl, slvr = forward(alpha)
    u,v = split(slvr.U)

    #J = (mdl.u_std**(-2)*(u-mdl.u_obs)**2 + mdl.v_std**(-2)*(v-mdl.v_obs)**2)*slvr.dObs
    J = inner(slvr.U,slvr.U)*slvr.dIce

    grad_alpha = grad(slvr.alpha)
    grad_alpha_ = project(grad_alpha, mdl.RT)
    lap_alpha = div(grad_alpha_)
    lambda_a = mdl.param['rc_inv'][0]
    delta_a = mdl.param['rc_inv'][2]
    reg_a = lambda_a * slvr.alpha - delta_a*lap_alpha
    reg_a_bndry = delta_a*inner(grad_alpha,mdl.nm)
    J_reg_alpha = inner(reg_a,reg_a)*slvr.dIce + inner(reg_a_bndry,reg_a_bndry)*slvr.ds

    #J += J_reg_alpha
    return assemble(J)

#minconv = taylor_test(J_test, cc, J_val, dJ, HJm = ddJ, seed = 0.1, size = 5)



minconv = taylor_test(J_test, cc, J_val, dJ, seed = 0.01, size = 4)

# direction = interpolate(Constant(1), alpha0.function_space())
# ddJ( direction)
import sys;  sys.exit(0)
