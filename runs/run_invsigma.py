# For fenics_ice copyright information see ACKNOWLEDGEMENTS in the fenics_ice
# root directory

# This file is part of fenics_ice.
#
# fenics_ice is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# fenics_ice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

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

def run_invsigma(config_file):
    """Compute control sigma values from eigendecomposition"""

    # Read run config file
    params = ConfigParser(config_file)
    log = inout.setup_logging(params)
    inout.log_preamble("inv sigma", params)

    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    eigendir = outdir
    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file
    threshlam = params.eigendec.eigenvalue_thresh

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model (only need alpha & beta though)
    mdl = model.model(mesh, input_data, params, init_fields=False)

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

    # test, trial = TestFunction(space), TrialFunction(space)
    # mass = assemble(inner(test, trial)*dx)
    # mass_solver = KrylovSolver("cg", "sor")
    # mass_solver.parameters.update({"absolute_tolerance": 1.0e-32,
    #                                "relative_tolerance": 1.0e-14})
    # mass_solver.set_operator(mass)

    # Load the eigenvalues
    with open(os.path.join(eigendir, lamfile), 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)

    # Read in the eigenvectors and check they are normalised
    # w.r.t. the prior (i.e. the B matrix in our GHEP)
    eps = params.constants.float_eps
    W = np.zeros((x.vector().size(), nlam))
    with HDF5File(MPI.comm_world,
                  os.path.join(eigendir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            hdf5data.read(x, f'v/vector_{i}')
            v = x.vector().get_local()
            reg_op.action(x.vector(), y.vector())
            tmp = y.vector().get_local()
            norm_in_prior = np.sqrt(np.dot(v, tmp))
            assert (abs(norm_in_prior - 1.0) < eps)
            W[:, i] = v

    # Which eigenvalues are larger than our threshold?
    pind = np.flatnonzero(lam > threshlam)
    lam = lam[pind]
    W = W[:, pind]

    D = np.diag(lam / (lam + 1))

    sigma_vector = np.zeros(space.dim())
    sigma_prior_vector = np.zeros(space.dim())
    ivec = np.zeros(space.dim())

    neg_flag = 0

    # Isaac Eq. 20
    # P2 = prior
    # P1 = WDW
    # Note - don't think we're considering the cross terms
    # in the posterior covariance.
    for j in range(sigma_vector.size):

        ivec.fill(0)
        ivec[j] = 1.0
        y.vector().set_local(ivec)
        y.vector().apply('insert')

        tmp1 = np.dot(W.T, ivec)  # take the ith row from W
        tmp2 = np.dot(D, tmp1)  # just a vector-vector product
        P1 = np.dot(W, tmp2)

        reg_op.inv_action(y.vector(), x.vector())
        P2 = x.vector().get_local()

        P = P2-P1
        dprod = np.dot(ivec, P)
        dprod_prior = np.dot(ivec, P2)

        if dprod < 0:
            log.warning(f'WARNING: Negative Sigma: {dprod}')
            log.warning('Setting as Zero and Continuing.')
            neg_flag = 1
            continue

        sigma_vector[j] = np.sqrt(dprod)
        sigma_prior_vector[j] = np.sqrt(dprod_prior)


    # For testing - whole thing at once:
    # wdw = (np.matrix(W) * np.matrix(D) * np.matrix(W).T)
    # wdw[:,0] == P1 for j = 0

    if neg_flag:
        log.warning('Negative value(s) of sigma encountered.'
                    'Examine the range of eigenvalues and check if '
                    'the threshlam paramater is set appropriately.')

    # Write values to vectors
    sigma.vector().set_local(sigma_vector)
    sigma.vector().apply('insert')

    sigma_prior.vector().set_local(sigma_prior_vector)
    sigma_prior.vector().apply('insert')

    # Write sigma & sigma_prior to files
    sigma_var_name = "_".join((cntrl.name(), "sigma"))
    sigma_prior_var_name = "_".join((cntrl.name(), "sigma_prior"))

    sigma.rename(sigma_var_name, "")
    sigma_prior.rename(sigma_prior_var_name, "")

    inout.write_variable(sigma, params,
                         name=sigma_var_name)
    inout.write_variable(sigma_prior, params,
                         name=sigma_prior_var_name)

    mdl.cntrl_sigma = sigma
    mdl.cntrl_sigma_prior = sigma_prior
    return mdl

if __name__ == "__main__":
    stop_annotating()

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_invsigma(sys.argv[1])
