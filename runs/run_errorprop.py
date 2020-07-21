import sys
import os
from pathlib import Path

from dolfin import *
from tlm_adjoint import *

from fenics_ice import model, solver, prior, inout
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

    assert MPI.size(MPI.comm_world) == 1, "Run this stage in serial!"

    # Read run config file
    params = ConfigParser(config_file)
    log = inout.setup_logging(params)
    inout.log_preamble("errorprop", params)

    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file
    threshlam = params.eigendec.eigenvalue_thresh
    dqoi_h5file = params.io.dqoi_h5file

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model
    mdl = model.model(mesh, input_data, params)

    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()


    # Regularization operator using inversion delta/gamma values
    # TODO - this won't handle dual inversion case
    if params.inversion.alpha_active:
        delta = params.inversion.delta_alpha
        gamma = params.inversion.gamma_alpha
        cntrl = mdl.alpha
    elif params.inversion.beta_active:
        delta = params.inversion.delta_beta
        gamma = params.inversion.gamma_beta
        cntrl = mdl.beta

    reg_op = prior.laplacian(delta, gamma, cntrl.function_space())

    space = cntrl.function_space()
    x, y, z = [Function(space) for i in range(3)]


    # TODO: not convinced this does anything at present
    # Was used to test that eigenvectors are prior inverse orthogonal
    # test, trial = TestFunction(space), TrialFunction(space)
    # mass = assemble(inner(test,trial)*dx)
    # mass_solver = KrylovSolver("cg", "sor")
    # mass_solver.parameters.update({"absolute_tolerance":1.0e-32,
    #                            "relative_tolerance":1.0e-14})
    # mass_solver.set_operator(mass)


    # Loads eigenvalues from file
    with open(os.path.join(outdir, lamfile), 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)

    # and eigenvectors from .h5 file
    eps = params.constants.float_eps
    W = np.zeros((x.vector().size(),nlam))

    with HDF5File(MPI.comm_world, os.path.join(outdir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            hdf5data.read(x, f'v/vector_{i}')
            v = x.vector().get_local()
            reg_op.action(x.vector(), y.vector())

            tmp = y.vector().get_local()
            norm_in_prior = np.sqrt(np.dot(v,tmp))
            assert (abs(norm_in_prior - 1.0) < eps)
            W[:,i] = v



    # Take only the largest eigenvalues
    pind = np.flatnonzero(lam>threshlam)
    lam = lam[pind]
    W = W[:,pind]

    D = np.diag(lam / (lam + 1)) #D_r Isaac 20

    # File containing dQoi_dCntrl (i.e. Jacobian of parameter to observable (Qoi))
    hdf5data = HDF5File(MPI.comm_world, os.path.join(outdir, dqoi_h5file), 'r')

    dQ_cntrl = Function(space)

    run_length = params.time.run_length
    num_sens = params.time.num_sens
    t_sens = np.flip(np.linspace(run_length, 0, num_sens))
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

    # This simplifies testing - is it OK? Should we hold all data in the solver object?
    mdl.Q_sigma = sigma
    mdl.Q_sigma_prior = sigma_prior
    mdl.t_sens = t_sens
    return mdl

if __name__ == "__main__":
    stop_annotating()

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_errorprop(sys.argv[1])
