import sys
import os
import argparse
from dolfin import *
from tlm_adjoint_fenics import *

import model
import solver
import mesh as fice_mesh
from config import *

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
import time
import datetime
import pickle
from IPython import embed
import prior

#import git

def run_sample(config_file):
    """
    Run the inversion part of the simulation
    """

#    repo = git.Repo(__file__, search_parent_directories=True)
#    branch = repo.active_branch.name
#    sha = repo.head.object.hexsha[:7]
#    print("=============== Fenics Ice ===============")
#    print("==   git branch  : %s" % branch)
#    print("==   commit hash : %s" % sha)
#    print("==========================================")

    #Read run config file
    params = ConfigParser(config_file)

    dd = params.io.input_dir

    # Determine Mesh

    mesh = fice_mesh.get_mesh(params)
    data_mesh = fice_mesh.get_data_mesh(params)

    # Define Function Spaces
    M = FunctionSpace(data_mesh, 'DG', 0)
    Q = FunctionSpace(data_mesh, 'Lagrange', 1)

    # Make necessary modification for periodic bc
    if params.mesh.periodic_bc:
        Qp = fice_mesh.setup_periodic_bc(params, data_mesh)
    else:
        Qp = Q

    data_mask = fice_mesh.get_data_mask(params, M)

    bed = Function(Q,os.path.join(dd,'bed.xml'))

    thick = Function(M,os.path.join(dd,'thick.xml'))
    u_obs = Function(M,os.path.join(dd,'u_obs.xml'))
    v_obs = Function(M,os.path.join(dd,'v_obs.xml'))
    u_std = Function(M,os.path.join(dd,'u_std.xml'))
    v_std = Function(M,os.path.join(dd,'v_std.xml'))
    mask_vel = Function(M,os.path.join(dd,'mask_vel.xml'))
    Bglen = Function(M,os.path.join(dd,'Bglen.xml'))
    bmelt = Function(M,os.path.join(dd,'bmelt.xml'))
    smb = Function(M,os.path.join(dd,'smb.xml'))

    pts_lengthscale = params.obs.pts_len

    mdl = model.model(mesh,data_mask, params)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(data_mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std, pts_lengthscale)
    mdl.init_lat_dirichletbc()
    mdl.init_bmelt(bmelt)
    mdl.init_smb(smb)
    mdl.label_domain()

    mdl.gen_alpha()
    mdl.init_beta(mdl.bglen_to_beta(Bglen))            #Comment to use uniform Bglen

    #Next line will output the initial guess for alpha fed into the inversion
    #File(os.path.join(outdir,'alpha_initguess.pvd')) << mdl.alpha

    #Inversion
    slvr = solver.ssa_solver(mdl)

    mesh = slvr.mesh
    Qp = slvr.Q

# ------------------------------------------------------

    delta = params.inversion.delta_alpha
    gamma = params.inversion.gamma_alpha
    reg_op = prior.laplacian(delta,gamma,Qp)

    x = Function(Qp, name="x")
    x =project(Expression("exp(x[0]) * exp(x[1])",degree=1),Qp)
    reg_op.sample(x.vector())

#    xpts    = mesh.coordinates()[:,0]
#    ypts    = mesh.coordinates()[:,1]
#    t    = mesh.cells()
#    fig = plt.figure(figsize=(10,5))
#    ax = fig.add_subplot(1,1,1)
#    v    = x.compute_vertex_values(mesh)
#    minv = np.min(v)
#    maxv = np.max(v)
#    levels = np.linspace(minv,maxv,20)
#    ticks = np.linspace(minv,maxv,3)
#    tick_options = {'axis':'both','which':'both','bottom':False,
#                    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}
#    ax.tick_params(**tick_options)
#    ax.text(0.05, 0.95, 'a', transform=ax.transAxes,
#            fontsize=13, fontweight='bold', va='top')
#    c = ax.tricontourf(xpts, ypts, t, v, levels = levels, cmap='bwr')
#    cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal")
#    plt.tight_layout(2.0)
#    plt.savefig(os.path.join(params.io.output_dir, 'sample_prior.pdf'))

    slvr.lambda_a = 0.0
    slvr.delta_a = 0.0
    slvr.delta_b = 0.0
    slvr.gamma_a = 0.0
    slvr.gamma_b = 0.0
    J = slvr.forward_alpha(x) 
    print(J.value())
    lambda_a = 1.0
    delta_a = params.inversion.delta_alpha
    delta_b = params.inversion.delta_beta
    gamma_a = params.inversion.gamma_alpha
    gamma_b = params.inversion.gamma_beta
    slvr.save_ts_zero()
    slvr.timestep(save=0,adjoint_flag=0, qoi_func=slvr.comp_Q_h2 )
    Q=slvr.comp_Q_h2(verbose=True)
    embed()


if __name__ == "__main__":
    stop_annotating()
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_sample(sys.argv[1])
