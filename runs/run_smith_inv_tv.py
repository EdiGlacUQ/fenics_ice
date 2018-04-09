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

#ss: flag to determine where to load domain data
#   1: Load directly from source files, alpha and beta are best guesses
#   0: Load from previous inversion. alpha and beta have been inverted for

ss = 1

if ss == 1:
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
            'rc_inv': [1.0, 1e-2, 1e-12, 5e4, 5e6],
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
    #mdl.init_alpha(Constant(sqrt(20000))) #Initialize using uniform alpha
    mdl.init_beta(sqrt(B_mod))            #Comment to use uniform Bglen
    mdl.label_domain()


else:
    #Load Data
    dd = './output_smith_inv/'
    ff = './output_smith_forward/'

    param = pickle.load( open( ''.join([ff,'param.p']), "rb" ) )
    param['outdir'] = './output_smith_analysis/'

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



slvr = solver.ssa_solver(mdl)
cc = Control(slvr.alpha)

slvr.taylor_ver_inv(slvr.alpha)
dJ = compute_gradient(Functional(slvr.J_inv), cc, forget = False)
ddJ = hessian(Functional(slvr.J_inv), cc)

minconv = taylor_test(slvr.taylor_ver_inv, cc, assemble(slvr.J_inv), dJ, HJm = ddJ, seed = 1e-1, size = 3)
