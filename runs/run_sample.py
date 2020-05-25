#!/usr/bin/env python
import sys

import os
import argparse
from fenics import *
from tlm_adjoint_fenics import *
import pickle
from IPython import embed

from fenics_ice import model, solver, prior, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser

import datetime
import numpy as np
import matplotlib as mpl
#mpl.use("Agg")
import matplotlib.pyplot as plt
import logging as log
from fenics_ice import mesh as fice_mesh

def run_sample(config_file):

    #Read run config file
    params = ConfigParser(config_file)
    inout.setup_logging(params)

    log.info("=======================================")
    log.info("=== RUNNING SAMPLING SCRIPT ==")
    log.info("=======================================")

    dd = params.io.input_dir
    outdir = params.io.output_dir

    #Ice only mesh
    dd = params.io.input_dir

    # Determine Mesh (1. create ismip or 2. from file)
    if params.mesh.nx:
        mesh = fice_mesh.create_ismip_mesh(params)
    else:
        mesh = fice_mesh.get_mesh(params)

    data_mesh = fice_mesh.get_data_mesh(params)

    # Define Function Spaces
    M = FunctionSpace(data_mesh, 'DG', 0)
    Q = FunctionSpace(data_mesh, 'Lagrange', 1)

    # Make necessary modification for periodic bc
    if params.mesh.periodic_bc:
        Qp = fice_mesh.get_periodic_space(params, data_mesh, dim=1)
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
    #Add random noise to Beta field iff we're inverting for it
    mdl.init_beta(mdl.bglen_to_beta(Bglen), params.inversion.beta_active)

    #Setup our solver object
    slvr = solver.ssa_solver(mdl)

    mesh = slvr.mesh
    Qp = slvr.Qp

# ------------------------------------------------------

    delta = params.inversion.delta_alpha
    gamma = params.inversion.gamma_alpha
    reg_op = prior.laplacian(delta,gamma,Qp)
    alpha = mdl.alpha

    x = Function(Qp, name="x")
# 1000.0 + 1000.0*np.sin(self.tiles*w*xx)*np.sin(self.tiles*w*yy)    
# Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta * t',degree = 2, alpha=alpha, beta=beta, t=0)    
#    B2exp = Expression('(1000.0 + 1000.0*sin(2*pi/L*x[0])*sin(2*pi/L*x[1]))**.5',degree=1,L=params.mesh.length)
    B2exp = Expression('sqrt(1000+1000*sin(2*pi/L*x[0])*sin(2*pi/L*x[1]))',degree=1,L=params.mesh.length)
    alpha = project(B2exp,Qp)
    reg_op.sample(x.vector())

    x.vector()[:] += alpha.vector()[:]
    slvr.zero_inv_params()
    J = slvr.forward_alpha(x) 
    print(J.value())

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
#    plt.show()
    
    slvr.alpha=x
    slvr.save_ts_zero()
    #Q2 = slvr.timestep(save=1,adjoint_flag=0,cost_flag=1,qoi_func=slvr.comp_Q_h2 )
    #for i in range(params.time.num_sens):
    #    print(Q2[i].value())

if __name__ == "__main__":
    stop_annotating()
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_sample(sys.argv[1])
