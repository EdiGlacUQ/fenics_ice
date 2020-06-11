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

def run_invsigma(config_file):
    #Read run config file
    params = ConfigParser(config_file)
    log = inout.setup_logging(params)
    inout.log_git_info()

    log.info("=======================================")
    log.info("========== RUNNING INV SIGMA ==========")
    log.info("=======================================")

    inout.print_config(params)

    dd = params.io.input_dir
    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    data_file = params.io.data_file
    input_data = inout.InputData(Path(dd) / data_file)

    eigendir = outdir
    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file
    threshlam = params.eigendec.eigenvalue_thresh

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model (only need alpha & beta though)
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

    space = cntrl.function_space()

    sigma, sigma_prior, x, y, z = [Function(space) for i in range(5)]

    reg_op = prior.laplacian(delta, gamma, space)

    test, trial = TestFunction(space), TrialFunction(space)
    mass = assemble(inner(test,trial)*dx)
    mass_solver = KrylovSolver("cg", "sor")
    mass_solver.parameters.update({"absolute_tolerance":1.0e-32,
                               "relative_tolerance":1.0e-14})
    mass_solver.set_operator(mass)


    with open(os.path.join(eigendir, lamfile), 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)


    W = np.zeros((x.vector().size(),nlam))
    with HDF5File(MPI.comm_world, os.path.join(eigendir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            hdf5data.read(x, f'v/vector_{i}')
            v = x.vector().get_local()
            reg_op.action(x.vector(), y.vector())
            tmp = y.vector().get_local()
            sc = np.sqrt(np.dot(v,tmp))
            W[:,i] = v/sc


    pind = np.flatnonzero(lam>threshlam)
    lam = lam[pind]
    W = W[:,pind]

    D = np.diag(lam / (lam + 1))


    sigma_vector = np.zeros(space.dim())
    sigma_prior_vector = np.zeros(space.dim())
    ivec = np.zeros(space.dim())

    neg_flag = 0

    for j in range(sigma_vector.size):

        ivec.fill(0)
        ivec[j] = 1.0
        y.vector().set_local(ivec)
        y.vector().apply('insert')

        tmp1 = np.dot(W.T,ivec)
        tmp2 = np.dot(D,tmp1 )
        P1 = np.dot(W,tmp2)

        reg_op.inv_action(y.vector(),x.vector())
        P2 = x.vector().get_local()

        P = P2-P1
        dprod = np.dot(ivec, P)
        dprod_prior = np.dot(ivec, P2)


        if dprod < 0:
            log.warning(f'WARNING: Negative Sigma: {dprod}')
            log.warning(f'Setting as Zero and Continuing.')
            neg_flag = 1
            continue

        sigma_vector[j] = np.sqrt(dprod)
        sigma_prior_vector[j] = np.sqrt(dprod_prior)


    if neg_flag:
        log.warning('Negative value(s) of sigma encountered. Examine the range of eigenvalues and check if the threshlam paramater is set appropriately.')
    
    sigma.vector().set_local(sigma_vector)
    sigma.vector().apply('insert')

    sigma_prior.vector().set_local(sigma_prior_vector)
    sigma_prior.vector().apply('insert')

    vtkfile = File(os.path.join(outdir,'{0}_sigma.pvd'.format(cntrl.name()) ))
    xmlfile = File(os.path.join(outdir,'{0}_sigma.xml'.format(cntrl.name()) ))
    vtkfile << sigma
    xmlfile << sigma

    vtkfile = File(os.path.join(outdir,'{0}_sigma_prior.pvd'.format(cntrl.name()) ))
    xmlfile = File(os.path.join(outdir,'{0}_sigma_prior.xml'.format(cntrl.name()) ))
    vtkfile << sigma_prior
    xmlfile << sigma_prior

if __name__ == "__main__":
    stop_annotating()

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_invsigma(sys.argv[1])
