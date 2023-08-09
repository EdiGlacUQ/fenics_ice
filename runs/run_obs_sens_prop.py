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

from fenics_ice.backend import Function, HDF5File
from tlm_adjoint import reset_manager, set_manager, stop_manager, \
        configure_tlm, function_tlm, restore_manager,\
        EquationManager, start_manager

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import mpi4py.MPI as MPI  # noqa: N817
from pathlib import Path
import pickle
import numpy as np
import sys

from fenics_ice import model, solver, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
from ufl import split
from fenics_ice.solver import Amat_obs_action

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from IPython import embed
from scipy.sparse import spdiags

@restore_manager
def compute_tau(forward, u, m, dm):
    # this block of code will do the "forward" (calculation of velocity and cost function) once
    # and then find the jacobian of u in the direction needed
    set_manager(EquationManager(cp_method="none", cp_parameters={}))
    stop_manager()

    start_manager(tlm=True)
    configure_tlm((m, dm))
    forward(m)
    return function_tlm(u, (m, dm))        


def run_obs_sens_prop(config_file):

    # Read run config file
    params = ConfigParser(config_file)

    # Setup logging
    inout.setup_logging(params)
    inout.log_preamble("obssensprop", params)

    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    # Eigen decomposition params
    phase_eigen = params.eigendec.phase_name
    phase_suffix_e = params.eigendec.phase_suffix
    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file

    # Qoi forward params
    phase_time = params.time.phase_name
    phase_suffix_qoi = params.time.phase_suffix
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
    slvr = solver.ssa_solver(mdl, mixed_space=params.inversion.dual, obs_sensitivity=True)

    # from errorprop -- this variable is not used in that fn
    #cntrl = slvr.get_control()[0]

    cntrl = slvr.get_control()
    space = slvr.get_control_space()
    slvr.forward(cntrl)

    # Regularization operator using inversion delta/gamma values
    Prior = mdl.get_prior()
    reg_op = Prior(slvr, space)

    # Loads eigenvalues from file
    outdir_e = Path(outdir)/phase_eigen/phase_suffix_e
    with open(outdir_e/lamfile, 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)

    # Check if eigendecomposition successfully produced num_eig
    # or if some are NaN
    if np.any(np.isnan(lam)):
        raise RuntimeError("NaN eigenvalue(s)")

    # and eigenvectors from .h5 file
    eps = params.constants.float_eps
    W = []
    with HDF5File(MPI.COMM_WORLD, str(outdir_e/vecfile), 'r') as hdf5data:
        for i in range(nlam):
            w = Function(space)
            hdf5data.read(w, f'v/vector_{i}')

            # Test squared norm in prior == 1.0
            B_inv_w = Function(space, space_type="conjugate_dual")
            reg_op.action(w.vector(), B_inv_w.vector())
            norm_sq_in_prior = w.vector().inner(B_inv_w.vector())
            assert (abs(norm_sq_in_prior - 1.0) < eps)
            del B_inv_w

            W.append(w)

    D = np.diag(lam / (lam + 1))  # D_r Isaac 20

    # File containing dQoi_dCntrl (i.e. Jacobian of parameter to observable (Qoi))
    outdir_qoi = Path(outdir)/phase_time/phase_suffix_qoi
    hdf5data = HDF5File(MPI.COMM_WORLD, str(outdir_qoi/dqoi_h5file), 'r')

    dQ_cntrl = Function(space, space_type="conjugate_dual")

    run_length = params.time.run_length
    num_sens = params.time.num_sens
    t_sens = np.flip(np.linspace(run_length, 0, num_sens))

    assert hasattr(slvr,'_cached_Amat_vars'),\
        "Attempt to retrieve Amat from solver type without first caching"
    (P, u_std_local, v_std_local, interp_space) = \
                slvr._cached_Amat_vars

    Ru = spdiags(1.0 / (u_std_local ** 2),
                              0, P.shape[0], P.shape[0])
    Rv = spdiags(1.0 / (v_std_local ** 2),
                              0, P.shape[0], P.shape[0])

    dObs = []

    for j in range(num_sens):
        hdf5data.read(dQ_cntrl, f'dQd{cntrl[0].name()}/vector_{j}')

        tmp1 = np.asarray([w.vector().inner(dQ_cntrl.vector()) for w in W])
        tmp2 = np.dot(D, tmp1)

        P1 = Function(space)
        for tmp, w in zip(tmp2, W):
            P1.vector().axpy(tmp, w.vector())

        P2 = Function(space)
        reg_op.inv_action(dQ_cntrl.vector(), P2.vector())

        P3 = Function(space)
        P3.vector()[:] = P2.vector()[:] - P1.vector()[:]
        

    # retaining the following code causes a tlm_adjoint runtime error in the 2nd execution of the loop 
    # so the code is packaged within a decorated function
       # reset_manager()
       # stop_manager(tlm=False)
       # configure_tlm((cntrl, P3))
       # slvr.forward(cntrl)
       # tau = function_tlm(slvr.U, (cntrl, P3))

        tau = compute_tau(slvr.forward, slvr.U, cntrl, P3)

    # tau is in the space of U (right?)
        tauu, tauv = split(tau)

    # this block of code then implements -A.tau

        dobs = Amat_obs_action(P, Ru, tauu, interp_space) + \
               Amat_obs_action(P, Rv, tauv, interp_space)
        dObs.append(dobs)
               
    # Look at the last sampled time and check how sigma QoI converges
    # with addition of more eigenvectors

    # Save plots in diagnostics
    # phase_err = params.error_prop.phase_name
    # phase_suffix_err = params.error_prop.phase_suffix
    # diag_dir = Path(params.io.diagnostics_dir)/phase_err/phase_suffix_err
    # outdir_err = Path(params.io.output_dir)/phase_err/phase_suffix_err


    #with open( os.path.join(outdir_err, sigma_file), "wb" ) as sigfile:
    #    pickle.dump( [sigma, t_sens], sigfile)
    #with open( os.path.join(outdir_err, sigma_prior_file), "wb" ) as sigpfile:
    #    pickle.dump( [sigma_prior, t_sens], sigpfile)



if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_obs_sens_prop(sys.argv[1])
