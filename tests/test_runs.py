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

import pytest
import numpy as np
from runs import run_inv, run_forward, run_eigendec, run_errorprop, run_invsigma
from tlm_adjoint import *
from fenics import norm
from fenics_ice import config
from pathlib import Path
from mpi4py import MPI

def EQReset():
    """Take care of tlm_adjoint EquationManager"""
    # This prevents checkpointing errors when these run phases
    # are tested after the stuff in test_model.py
    reset_manager("memory")
    clear_caches()
    stop_manager()

@pytest.mark.dependency()
@pytest.mark.runs
def test_run_inversion(persistent_temp_model, monkeypatch):

    work_dir = persistent_temp_model["work_dir"]
    toml_file = persistent_temp_model["toml_filename"]

    # Switch to the working directory
    monkeypatch.chdir(work_dir)

    # Get expected values from the toml file
    params = config.ConfigParser(toml_file, top_dir=work_dir)
    expected_cntrl_norm = params.testing.expected_cntrl_norm
    expected_J_inv = params.testing.expected_J_inv

    EQReset()

    # Run the thing
    mdl_out = run_inv.run_inv(toml_file)

    cntrl = mdl_out.solvers[0].get_control()[0]
    cntrl_norm = norm(cntrl.vector())

    J_inv = mdl_out.solvers[0].J_inv.value()

    pytest.check_float_result(cntrl_norm,
                              expected_cntrl_norm,
                              work_dir, 'expected_cntrl_norm')

    pytest.check_float_result(J_inv,
                              expected_J_inv,
                              work_dir, 'expected_J_inv')

    # Taylor verification
    alpha_active = mdl_out.params.inversion.alpha_active
    beta_active = mdl_out.params.inversion.beta_active

    if alpha_active:

        fwd_alpha = mdl_out.solvers[0].forward_alpha
        alpha = mdl_out.solvers[0].alpha

        min_order = taylor_test_tlm(fwd_alpha,
                                    alpha,
                                    tlm_order=1,
                                    seed=1.0e-6)                  # mpi-2 ice stream, serial ismipc
        print(f"min order alpha 1st order forward: {min_order}")  # 1.937, 2.0000

        min_order = taylor_test_tlm_adjoint(fwd_alpha,
                                            alpha,
                                            adjoint_order=1,
                                            seed=1.0e-6)
        print(f"min order alpha 1st order adjoint: {min_order}")  # 1.000, 1.0001

        min_order = taylor_test_tlm_adjoint(fwd_alpha,
                                            alpha,
                                            adjoint_order=2,
                                            seed=1.0e-6)
        print(f"min order alpha 2nd order adjoint: {min_order}")  # 1.000, 0.9998

    if beta_active:

        fwd_beta = mdl_out.solvers[0].forward_beta
        beta = mdl_out.solvers[0].beta

        min_order = taylor_test_tlm(fwd_beta,
                                    beta,
                                    tlm_order=1,
                                    seed=1.0e-6)
        print(f"min order beta 1st order forward: {min_order}")  # 1.9999, N/A

        min_order = taylor_test_tlm_adjoint(fwd_beta,
                                            beta,
                                            adjoint_order=1,
                                            seed=1.0e-6)
        print(f"min order beta 1st order adjoint: {min_order}")  # 1.0055, N/A

        min_order = taylor_test_tlm_adjoint(fwd_beta,
                                            beta,
                                            adjoint_order=2,
                                            seed=1.0e-6)
        print(f"min order beta 2nd order adjoint: {min_order}")  # 0.9999, N/A

@pytest.mark.dependency()
@pytest.mark.runs
def test_run_forward(existing_temp_model, monkeypatch, setup_deps, request):

    setup_deps.set_case_dependency(request, ["test_run_inversion"])

    work_dir = existing_temp_model["work_dir"]
    toml_file = existing_temp_model["toml_filename"]

    # Switch to the working directory
    monkeypatch.chdir(work_dir)

    # Get expected values from the toml file
    params = config.ConfigParser(toml_file, top_dir=work_dir)
    expected_delta_qoi = params.testing.expected_delta_qoi
    expected_u_norm = params.testing.expected_u_norm

    EQReset()

    mdl_out = run_forward.run_forward(toml_file)

    slvr = mdl_out.solvers[0]

    delta_qoi = slvr.Qval_ts[-1] - slvr.Qval_ts[0]
    u_norm = norm(slvr.U)

    pytest.check_float_result(delta_qoi,
                              expected_delta_qoi,
                              work_dir, 'expected_delta_qoi')

    pytest.check_float_result(u_norm,
                              expected_u_norm,
                              work_dir, 'expected_u_norm')


@pytest.mark.dependency()
@pytest.mark.runs
def test_run_eigendec(existing_temp_model, monkeypatch, setup_deps, request):

    setup_deps.set_case_dependency(request, ["test_run_inversion"])

    work_dir = existing_temp_model["work_dir"]
    toml_file = existing_temp_model["toml_filename"]

    # Switch to the working directory
    monkeypatch.chdir(work_dir)

    # Get expected values from the toml file
    params = config.ConfigParser(toml_file, top_dir=work_dir)
    expected_evals_sum = params.testing.expected_evals_sum
    expected_evec0_norm = params.testing.expected_evec0_norm

    EQReset()

    mdl_out = run_eigendec.run_eigendec(toml_file)

    slvr = mdl_out.solvers[0]

    evals_sum = np.sum(slvr.eigenvals)
    evec0_norm = norm(slvr.eigenfuncs[0])

    pytest.check_float_result(evals_sum,
                              expected_evals_sum,
                              work_dir, 'expected_evals_sum')
    pytest.check_float_result(evec0_norm,
                              expected_evec0_norm,
                              work_dir, 'expected_evec0_norm')

@pytest.mark.dependency()
@pytest.mark.runs
def test_run_errorprop(existing_temp_model, monkeypatch, setup_deps, request):

    setup_deps.set_case_dependency(request, ["test_run_eigendec", "test_run_forward"])

    work_dir = existing_temp_model["work_dir"]
    toml_file = existing_temp_model["toml_filename"]

    # Switch to the working directory
    monkeypatch.chdir(work_dir)

    # Get expected values from the toml file
    params = config.ConfigParser(toml_file, top_dir=work_dir)
    expected_Q_sigma = params.testing.expected_Q_sigma
    expected_Q_sigma_prior = params.testing.expected_Q_sigma_prior

    EQReset()

    mdl_out = run_errorprop.run_errorprop(toml_file)

    Q_sigma = mdl_out.Q_sigma[-1]
    Q_sigma_prior = mdl_out.Q_sigma_prior[-1]

    if pytest.parallel:
        tol = 1e-6
    else:
        tol = 1e-7

    pytest.check_float_result(Q_sigma,
                              expected_Q_sigma,
                              work_dir,
                              'expected_Q_sigma', tol=tol)
    pytest.check_float_result(Q_sigma_prior,
                              expected_Q_sigma_prior,
                              work_dir,
                              'expected_Q_sigma_prior', tol=tol)

@pytest.mark.dependency()
@pytest.mark.runs
def test_run_invsigma(existing_temp_model, monkeypatch, setup_deps, request):

    setup_deps.set_case_dependency(request, ["test_run_eigendec"])

    work_dir = existing_temp_model["work_dir"]
    toml_file = existing_temp_model["toml_filename"]

    # Switch to the working directory
    monkeypatch.chdir(work_dir)

    # Get expected values from the toml file
    params = config.ConfigParser(toml_file, top_dir=work_dir)
    expected_cntrl_sigma_norm = params.testing.expected_cntrl_sigma_norm
    expected_cntrl_sigma_prior_norm = params.testing.expected_cntrl_sigma_prior_norm

    EQReset()

    mdl_out = run_invsigma.run_invsigma(toml_file)

    cntrl_sigma_norm = norm(mdl_out.cntrl_sigma)
    cntrl_sigma_prior_norm = norm(mdl_out.cntrl_sigma_prior)

    if pytest.parallel:
        tol = 1e-6
    else:
        tol = 1e-7

    pytest.check_float_result(cntrl_sigma_norm,
                              expected_cntrl_sigma_norm,
                              work_dir,
                              "expected_cntrl_sigma_norm", tol=tol)

    pytest.check_float_result(cntrl_sigma_prior_norm,
                              expected_cntrl_sigma_prior_norm,
                              work_dir,
                              "expected_cntrl_sigma_prior_norm", tol=tol)
