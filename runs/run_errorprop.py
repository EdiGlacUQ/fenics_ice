import sys

import os
import argparse
from dolfin import *
from tlm_adjoint import *

from fenics_ice import model, solver, prior
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
import fenics_ice.fenics_util as fu

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import pickle
from petsc4py import PETSc
from IPython import embed


def run_errorprop(config_file):


    print("=======================================")
    print("=== RUNNING ERROR PROPAGATION PHASE ===")
    print("=======================================")

    # eigendir
    # lamfile
    # vecfile
    # threshlam

    #Read run config file
    params = ConfigParser(config_file)

    dd = params.io.input_dir
    outdir = params.io.output_dir
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


    mdl = model.model(mesh, mask, params)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
    mdl.init_lat_dirichletbc()
    mdl.label_domain()
    mdl.init_alpha(alpha)



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
    mass = assemble(inner(test,trial)*dx)
    mass_solver = KrylovSolver("cg", "sor")
    mass_solver.parameters.update({"absolute_tolerance":1.0e-32,
                               "relative_tolerance":1.0e-14})
    mass_solver.set_operator(mass)


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

    D = np.diag(lam / (lam + 1)) #D_r Isaac 20

    hdf5data = HDF5File(MPI.comm_world, os.path.join(outdir, dqoi_h5file), 'r')

    dQ_cntrl = Function(space)

    run_length = params.time.run_length
    num_sens = params.time.num_sens
    t_sens = run_length if num_sens == 1 else np.linspace(0, run_length,num_sens)
    sigma = np.zeros(num_sens)
    sigma_prior = np.zeros(num_sens)


    for j in range(num_sens):
        hdf5data.read(dQ_cntrl, f'dQ/vector_{j}')

        #TODO - is a mass matrix operation required here?
        #qd_cntrl - should be gradients
        tmp1 = np.dot(W.T,dQ_cntrl.vector().get_local())
        tmp2 = np.dot(D,tmp1 )
        P1 = np.dot(W,tmp2)

        reg_op.inv_action(dQ_cntrl.vector(),x.vector())
        P2 = x.vector().get_local()

        P = P2-P1
        variance = np.dot(dQ_cntrl.vector().get_local(), P)
        sigma[j] = np.sqrt(variance)

        #Prior only
        variance_prior = np.dot(dQ_cntrl.vector().get_local(), P2)
        sigma_prior[j] = np.sqrt(variance_prior)


    #Test that eigenvectors are prior inverse orthogonal
    # y.vector().set_local(W[:,398])
    # y.vector().apply('insert')
    # reg_op.action(y.vector(), x.vector())
    # #mass.mult(x.vector(),z.vector())
    # q = np.dot(y.vector().get_local(),x.vector().get_local())


    #Output model variables in ParaView+Fenics friendly format
    sigma_file = params.io.sigma_file
    sigma_prior_file = params.io.sigma_prior_file
    pickle.dump( [sigma, t_sens], open( os.path.join(outdir,sigma_file), "wb" ) )
    pickle.dump( [sigma_prior, t_sens], open( os.path.join(outdir,sigma_prior_file), "wb" ) )


if __name__ == "__main__":
    stop_annotating()

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_errorprop(sys.argv[1])
