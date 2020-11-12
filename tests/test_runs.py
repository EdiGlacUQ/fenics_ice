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

    pytest.check_float_result(Q_sigma,
                              expected_Q_sigma,
                              work_dir,
                              'expected_Q_sigma', tol=1e-7)
    pytest.check_float_result(Q_sigma_prior,
                              expected_Q_sigma_prior,
                              work_dir,
                              'expected_Q_sigma_prior', tol=1e-7)

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

    pytest.check_float_result(cntrl_sigma_norm,
                              expected_cntrl_sigma_norm,
                              work_dir,
                              "expected_cntrl_sigma_norm", tol=1e-7)

    pytest.check_float_result(cntrl_sigma_prior_norm,
                              expected_cntrl_sigma_prior_norm,
                              work_dir,
                              "expected_cntrl_sigma_prior_norm", tol=1e-7)
