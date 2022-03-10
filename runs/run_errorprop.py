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

from fenics_ice.backend import Function, HDF5File, MPI

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import pickle

from fenics_ice import model, solver, prior, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def run_errorprop(config_file):

    # Read run config file
    params = ConfigParser(config_file)
    log = inout.setup_logging(params)
    inout.log_preamble("errorprop", params)

    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    phase_suffix_e = params.eigendec.phase_suffix
    phase_suffix_qoi = params.time.phase_suffix

    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file
    threshlam = params.eigendec.eigenvalue_thresh
    dqoi_h5file = params.io.dqoi_h5file

    if len(phase_suffix_e) > 0:
        lamfile = params.io.run_name + phase_suffix_e + '_eigvals.p'
        vecfile = params.io.run_name + phase_suffix_e + '_vr.h5'
    if len(phase_suffix_qoi) > 0:
        dqoi_h5file = params.io.run_name + phase_suffix_qoi + '_dQ_ts.h5'

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model
    mdl = model.model(mesh, input_data, params)

    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()
    mdl.bglen_from_data(mask_only=True)

    # Setup our solver object
    slvr = solver.ssa_solver(mdl, mixed_space=params.inversion.dual)

    cntrl = slvr.get_control()[0]
    space = slvr.get_control_space()

    # Regularization operator using inversion delta/gamma values
    Prior = mdl.get_prior()
    reg_op = Prior(slvr, space)

    x, y, z = [Function(space) for i in range(3)]

    # Loads eigenvalues from file
    with open(os.path.join(outdir, lamfile), 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)

    # Check if eigendecomposition successfully produced num_eig
    # or if some are NaN
    if np.any(np.isnan(lam)):
        nlam = np.argwhere(np.isnan(lam))[0][0]
        lam = lam[:nlam]

    # and eigenvectors from .h5 file
    eps = params.constants.float_eps
    W = []
    with HDF5File(MPI.comm_world, os.path.join(outdir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            w = Function(space)
            hdf5data.read(w, f'v/vector_{i}')

            # Test norm in prior == 1.0
            reg_op.action(w.vector(), y.vector())
            norm_in_prior = w.vector().inner(y.vector())
            assert (abs(norm_in_prior - 1.0) < eps)

            W.append(w)

    # take only the largest eigenvalues
    pind = np.flatnonzero(lam > threshlam)
    lam = lam[pind]
    nlam = len(lam)
    W = [W[i] for i in pind]

    D = np.diag(lam / (lam + 1))  # D_r Isaac 20

    # File containing dQoi_dCntrl (i.e. Jacobian of parameter to observable (Qoi))
    hdf5data = HDF5File(MPI.comm_world, os.path.join(outdir, dqoi_h5file), 'r')

    dQ_cntrl = Function(space)

    run_length = params.time.run_length
    num_sens = params.time.num_sens
    t_sens = np.flip(np.linspace(run_length, 0, num_sens))
    sigma = np.zeros(num_sens)
    sigma_prior = np.zeros(num_sens)

    for j in range(num_sens):
        hdf5data.read(dQ_cntrl, f'dQd{cntrl.name()}/vector_{j}')

        # TODO - is a mass matrix operation required here?
        # qd_cntrl - should be gradients
        tmp1 = np.asarray([w.vector().inner(dQ_cntrl.vector()) for w in W])
        tmp2 = np.dot(D, tmp1)

        P1 = Function(space)
        for tmp, w in zip(tmp2, W):
            P1.vector().axpy(tmp, w.vector())

        reg_op.inv_action(dQ_cntrl.vector(), x.vector())
        P2 = x  # .vector().get_local()

        P_vec = P2.vector() - P1.vector()

        variance = P_vec.inner(dQ_cntrl.vector())
        sigma[j] = np.sqrt(variance)

        # Prior only
        variance_prior = P2.vector().inner(dQ_cntrl.vector())
        sigma_prior[j] = np.sqrt(variance_prior)

    # Look at the last sampled time and check how sigma QoI converges
    # with addition of more eigenvectors

    sigma_conv = []
    sigma_steps = []
    P1 = Function(space)

    # How many steps?
    conv_res = 100
    conv_int = int(np.ceil(nlam/conv_res))

    for i in range(0, nlam, conv_int):

        # Reuse tmp1/tmp2 from above because its the last sens
        for j in range(i, min(i+conv_int, nlam)):
            P1.vector().axpy(tmp2[j], W[j].vector())

        P_vec = P2.vector() - P1.vector()

        variance = P_vec.inner(dQ_cntrl.vector())
        sigma_conv.append(np.sqrt(variance))
        sigma_steps.append(min(i+conv_int, nlam))


    # if(MPI.comm_world.rank == 0):
    plt.semilogy(sigma_steps, sigma_conv)
    plt.title("Convergence of sigmaQoI")
    plt.ylabel("sigma QoI")
    plt.xlabel("Num eig")

    plt.savefig(os.path.join(params.io.output_dir,
                             "_".join((params.io.run_name,
                                       phase_suffix_qoi +
                                       "sigmaQoI_conv.pdf"))))
    plt.close()

    sigmaqoi_file = os.path.join(params.io.output_dir,
                                 "_".join((params.io.run_name,
                                           phase_suffix_qoi +
                                           "sigma_qoi_convergence.p")))

    with open(sigmaqoi_file, 'wb') as pfile:
        pickle.dump([sigma_steps, sigma_conv], pfile)

    # Test that eigenvectors are prior inverse orthogonal
    # y.vector().set_local(W[:,398])
    # y.vector().apply('insert')
    # reg_op.action(y.vector(), x.vector())
    # #mass.mult(x.vector(),z.vector())
    # q = np.dot(y.vector().get_local(),x.vector().get_local())

    # Output model variables in ParaView+Fenics friendly format
    sigma_file = params.io.sigma_file
    sigma_prior_file = params.io.sigma_prior_file

    if len(phase_suffix_qoi) > 0:
        sigma_file = params.io.run_name + phase_suffix_qoi + '_sigma.p'
        sigma_prior_file = params.io.run_name + phase_suffix_qoi + '_sigma_prior.p'

    with open( os.path.join(outdir, sigma_file), "wb" ) as sigfile:
        pickle.dump( [sigma, t_sens], sigfile)
    with open( os.path.join(outdir, sigma_prior_file), "wb" ) as sigpfile:
        pickle.dump( [sigma_prior, t_sens], sigpfile)

    # This simplifies testing - is it OK? Should we hold all data in the solver object?
    mdl.Q_sigma = sigma
    mdl.Q_sigma_prior = sigma_prior
    mdl.t_sens = t_sens
    return mdl


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_errorprop(sys.argv[1])
