import sys

import os
import argparse
from dolfin import *
from tlm_adjoint import *

from fenics_ice import model, solver, prior, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
import fenics_ice.fenics_util as fu
from numpy import random
from pathlib import Path



import matplotlib as mpl
#mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import pickle
from petsc4py import PETSc
from IPython import embed
import logging as log

def run_sample_post(config_file):

    #Read run config file
    params = ConfigParser(config_file)
    inout.setup_logging(params)

    log.info("=======================================")
    log.info("=== RUNNING POSTERIOR SAMPLE  PHASE ===")
    log.info("=======================================")

    dd = params.io.input_dir
    outdir = params.io.output_dir
    plotdir = Path(os.environ['FENICS_ICE_BASE_DIR']) / 'example_cases' / params.io.run_name / 'plots'
    eigendir = outdir
    lamfile = params.io.eigenvalue_file
    vecfile = 'vr.h5' #TODO unharcode
    threshlam = params.eigendec.eigenvalue_thresh
    dqoi_h5file = params.io.dqoi_h5file

    #Load Data
    mesh = Mesh(os.path.join(outdir,'mesh.xml'))

    #Set up Function spaces
    Q = FunctionSpace(mesh,'Lagrange',1)
    M = FunctionSpace(mesh,'DG',0)

    #Handle function with optional periodic boundary conditions
    if not params.mesh.periodic_bc:
       Qp = Q
       V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
    else:
       Qp = fice_mesh.get_periodic_space(params, mesh, dim=1)
       V = fice_mesh.get_periodic_space(params, mesh, dim=2)

    #Load fields
    U = Function(V,os.path.join(outdir,'U.xml'))

    alpha = Function(Qp,os.path.join(outdir,'alpha.xml'))
    beta = Function(Qp,os.path.join(outdir,'beta.xml'))
    bed = Function(Q,os.path.join(outdir,'bed.xml'))

    thick = Function(M,os.path.join(outdir,'thick.xml'))
    mask = Function(M,os.path.join(outdir,'mask.xml'))
    mask_vel = Function(M,os.path.join(outdir,'mask_vel.xml'))
    u_obs = Function(M,os.path.join(outdir,'u_obs.xml'))
    v_obs = Function(M,os.path.join(outdir,'v_obs.xml'))
    u_std = Function(M,os.path.join(outdir,'u_std.xml'))
    v_std = Function(M,os.path.join(outdir,'v_std.xml'))
    uv_obs = Function(M,os.path.join(outdir,'uv_obs.xml'))
    bmelt = Function(M)
    smb = Function(M)


    mdl = model.model(mesh, mask, params)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
    mdl.init_lat_dirichletbc()
    mdl.label_domain()
    mdl.init_alpha(alpha)
    mdl.init_bmelt(bmelt)
    mdl.init_smb(smb)



    #Regularization operator using inversion delta/gamma values
    #TODO - this won't handle dual inversion case
    if params.inversion.alpha_active:
        delta = params.inversion.delta_alpha
        gamma = params.inversion.gamma_alpha
        cntrl = alpha
    elif params.inversion.beta_active:
        delta = params.inversion.delta_beta
        gamma = params.inversion.gamma_beta
        cntrl = beta

    reg_op = prior.laplacian(delta, gamma, cntrl.function_space())

    space = cntrl.function_space()
    x, y, z = [Function(space) for i in range(3)]


    #TODO: not convinced this does anything at present
    #Was used to test that eigenvectors are prior inverse orthogonal
    test, trial = TestFunction(space), TrialFunction(space)
#    mass = assemble(inner(test,trial)*dx)
#    mass_solver = KrylovSolver("cg", "sor")
#    mass_solver.parameters.update({"absolute_tolerance":1.0e-32,
#                               "relative_tolerance":1.0e-14})
#    mass_solver.set_operator(mass)


    #Loads eigenvalues from slepceig_all.p
    with open(os.path.join(eigendir, lamfile), 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)

    #and eigenvectors from .h5 files
    W = np.zeros((x.vector().size(),nlam))
    with HDF5File(MPI.comm_world, os.path.join(eigendir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            hdf5data.read(x, f'v/vector_{i}')
            v = x.vector().get_local()
            reg_op.action(x.vector(), y.vector())

            tmp = y.vector().get_local()
            sc = np.sqrt(np.dot(v,tmp))
            #eigenvectors are scaled by something to do with the prior action...
            #sc <- isaac between Eqs. 17, 18 <- normalised
            W[:,i] = v/sc



    #Take only the largest eigenvalues
    pind = np.flatnonzero(lam>threshlam)
    lam = lam[pind]
    W = W[:,pind]
    D = np.diag(1 / np.sqrt(1+lam) - 1)
    
    x, y, z, a = [Function(space) for i in range(4)]
    
    shp = np.shape(z.vector().get_local())
    
    
    slvr = solver.ssa_solver(mdl)
        
    Qarray = np.zeros((params.time.num_sens,params.time.num_samples))
    
    for i in range(params.time.num_samples):
        np.random.seed()
        x.vector().set_local(random.normal(np.zeros(shp),  # N
                         np.ones(shp),shp))
        x.vector().apply("insert")
        
        reg_op.sqrt_action(x.vector(),y.vector())  # Gamma -1/2 N
        reg_op.sqrt_inv_action(x.vector(),z.vector())  # Gamma 1/2 N
        
        tmp1 = np.dot(W.T,y.vector().get_local())
        tmp2 = np.dot(D,tmp1)
        P1 = np.dot(W,tmp2)
        
        a.vector().set_local(z.vector().get_local() + P1)
        a.vector().apply("insert")
        a.vector()[:] += alpha.vector()[:]
        z.vector()[:] += alpha.vector()[:]
        
        slvr.alpha=x
        slvr.save_ts_zero()
        Q = slvr.timestep(save=1,adjoint_flag=0,cost_flag=1,qoi_func=slvr.comp_Q_h2 )
        for j in range(params.time.num_sens):
            Qarray[j,i] = Q[j].value()
            
    np.save(os.path.join(outdir,'sampling_results'),Qarray)
            

#    embed()
#    a.vector()[:] += a.vector()[:]**2
#    z.vector()[:] += z.vector()[:]**2
    
    
    
#    xpts    = mesh.coordinates()[:,0]
#    ypts    = mesh.coordinates()[:,1]
#    t    = mesh.cells()
#    fig = plt.figure(figsize=(10,5))
#    ax = fig.add_subplot(1,2,1)
#    v    = z.compute_vertex_values(mesh)
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
#    
#    ax = fig.add_subplot(1,2,2)
#    v    = a.compute_vertex_values(mesh)
#    minv = np.min(v)
#    maxv = np.max(v)
#    levels = np.linspace(minv,maxv,20)
#    ticks = np.linspace(minv,maxv,3)
#    tick_options = {'axis':'both','which':'both','bottom':False,
#                    'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}
#    ax.tick_params(**tick_options)
#    ax.text(0.05, 0.95, 'b', transform=ax.transAxes,
#            fontsize=13, fontweight='bold', va='top')
#    c = ax.tricontourf(xpts, ypts, t, v, levels = levels, cmap='bwr')
#    cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal")
#    plt.tight_layout(2.0)
#    plt.show()
#    plt.savefig(os.path.join(plotdir, 'sample.pdf'))
    
#    D = np.diag(lam / (lam + 1)) #D_r Isaac 20
#
#    hdf5data = HDF5File(MPI.comm_world, os.path.join(outdir, dqoi_h5file), 'r')
#
#    dQ_cntrl = Function(space)
#
#    run_length = params.time.run_length
#    num_sens = params.time.num_sens
#    t_sens = run_length if num_sens == 1 else np.linspace(0, run_length,num_sens)
#    sigma = np.zeros(num_sens)
#    sigma_prior = np.zeros(num_sens)
#
#
#    for j in range(num_sens):
#        hdf5data.read(dQ_cntrl, f'dQ/vector_{j}')
#
#        #TODO - is a mass matrix operation required here?
#        #qd_cntrl - should be gradients
#        tmp1 = np.dot(W.T,dQ_cntrl.vector().get_local())
#        tmp2 = np.dot(D,tmp1 )
#        P1 = np.dot(W,tmp2)
#
#        reg_op.inv_action(dQ_cntrl.vector(),x.vector())
#        P2 = x.vector().get_local()
#
#        P = P2-P1
#        variance = np.dot(dQ_cntrl.vector().get_local(), P)
#        sigma[j] = np.sqrt(variance)
#
#        #Prior only
#        variance_prior = np.dot(dQ_cntrl.vector().get_local(), P2)
#        sigma_prior[j] = np.sqrt(variance_prior)







if __name__ == "__main__":
    stop_annotating()

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_sample_post(sys.argv[1])
