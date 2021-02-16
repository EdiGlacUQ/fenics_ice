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
import pickle
import numpy as np

from dolfin import *
from tlm_adjoint.fenics import *

from fenics_ice import model, solver, prior, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def run_invsigma(config_file):
    """Compute control sigma values from eigendecomposition"""

    comm = MPI.comm_world
    rank = comm.rank

    # Read run config file
    params = ConfigParser(config_file)

    # Setup logging
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
    mdl = model.model(mesh, input_data, params, init_fields=True)

    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()

    # Setup our solver object
    slvr = solver.ssa_solver(mdl, mixed_space=params.inversion.dual)

    cntrl = slvr.get_control()[0]
    space = slvr.get_control_space()

    sigma, sigma_prior, x, y, z = [Function(space) for i in range(5)]

    # Regularization operator using inversion delta/gamma values
    Prior = mdl.get_prior()
    reg_op = Prior(slvr, space)

    # Load the eigenvalues
    with open(os.path.join(eigendir, lamfile), 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)

    # Read in the eigenvectors and check they are normalised
    # w.r.t. the prior (i.e. the B matrix in our GHEP)
    eps = params.constants.float_eps
    W = []
    with HDF5File(comm,
                  os.path.join(eigendir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            w = Function(space)
            hdf5data.read(w, f'v/vector_{i}')

            # Test norm in prior == 1.0
            reg_op.action(w.vector(), y.vector())
            norm_in_prior = w.vector().inner(y.vector())
            assert (abs(norm_in_prior - 1.0) < eps)

            W.append(w)

    # Which eigenvalues are larger than our threshold?
    pind = np.flatnonzero(lam > threshlam)
    lam = lam[pind]
    W = [W[i] for i in pind]

    D = np.diag(lam / (lam + 1))

    neg_flag = 0

    # Isaac Eq. 20
    # P2 = prior
    # P1 = WDW
    # Note - don't think we're considering the cross terms
    # in the posterior covariance.
    # TODO - this isn't particularly well parallelised - can it be improved?
    for j in range(space.dim()):

        # Who owns this index?
        own_idx = y.vector().owns_index(j)
        ownership = np.where(comm.allgather(own_idx))[0]
        assert len(ownership) == 1
        idx_root  = ownership[0]

        y.vector().zero()
        y.vector().vec().setValue(j, 1.0)
        y.vector().apply('insert')

        tmp2 = np.asarray([D[i, i] * w.vector().vec().getValue(j) for i, w in enumerate(W)])
        tmp2 = comm.bcast(tmp2, root=idx_root)

        P1 = Function(space)
        for tmp, w in zip(tmp2, W):
            P1.vector().axpy(tmp, w.vector())

        reg_op.inv_action(y.vector(), x.vector())
        P2 = x

        P_vec = P2.vector() - P1.vector()

        dprod = comm.bcast(P_vec.vec().getValue(j), root=idx_root)
        dprod_prior = comm.bcast(P2.vector().vec().getValue(j), root=idx_root)

        if dprod < 0:
            log.warning(f'WARNING: Negative Sigma: {dprod}')
            log.warning('Setting as Zero and Continuing.')
            neg_flag = 1
            continue

        sigma.vector().vec().setValue(j, np.sqrt(dprod))
        sigma_prior.vector().vec().setValue(j, np.sqrt(dprod_prior))

    sigma.vector().apply("insert")
    sigma_prior.vector().apply("insert")

    # For testing - whole thing at once:
    # wdw = (np.matrix(W) * np.matrix(D) * np.matrix(W).T)
    # wdw[:,0] == P1 for j = 0

    if neg_flag:
        log.warning('Negative value(s) of sigma encountered.'
                    'Examine the range of eigenvalues and check if '
                    'the threshlam paramater is set appropriately.')

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
