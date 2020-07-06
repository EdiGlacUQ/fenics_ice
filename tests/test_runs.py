import pytest
import numpy as np
from runs import run_inv, run_forward, run_eigendec, run_errorprop, run_invsigma
from tlm_adjoint import *
from fenics import norm
from fenics_ice import config
from pathlib import Path
import pickle

pytest.temp_results = "/home/joe/sources/fenics_ice/tests/expected_values.txt"

def EQReset():
    """Take care of tlm_adjoint EquationManager"""
    # This prevents checkpointing errors when these run phases
    # are tested after the stuff in test_model.py
    reset_manager("memory")
    clear_caches()
    stop_manager()

@pytest.mark.dependency()
@pytest.mark.runs
@pytest.mark.benchmark()  # <- just run it once
def test_run_inv(persistent_temp_model, monkeypatch, benchmark):

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
    mdl_out = benchmark.pedantic(run_inv.run_inv,
                                 args=(toml_file,),
                                 rounds=1,
                                 warmup_rounds=0,
                                 iterations=1)

    cntrl = mdl_out.solvers[0].get_control()[0]
    cntrl_norm = np.linalg.norm(cntrl.vector()[:])

    J_inv = mdl_out.solvers[0].J_inv.value()

    # with open(pytest.temp_results, 'a') as output:
    #     output.write(f"{toml_file} - cntrl_norm - {cntrl_norm}\n")
    # temp_model["expected_cntrl_norm"] = cntrl_norm

    pytest.check_float_result(cntrl_norm, expected_cntrl_norm)
    pytest.check_float_result(J_inv, expected_J_inv)

@pytest.mark.dependency()
@pytest.mark.runs
@pytest.mark.benchmark()
def test_run_forward(existing_temp_model, monkeypatch, benchmark, setup_deps, request):

    setup_deps.set_case_dependency(request, ["test_run_inv"])

    work_dir = existing_temp_model["work_dir"]
    toml_file = existing_temp_model["toml_filename"]

    # Switch to the working directory
    monkeypatch.chdir(work_dir)

    # Get expected values from the toml file
    params = config.ConfigParser(toml_file, top_dir=work_dir)
    expected_delta_qoi = params.testing.expected_delta_qoi
    expected_u_norm = params.testing.expected_u_norm

    EQReset()

    mdl_out = benchmark.pedantic(run_forward.run_forward,
                                 args=(toml_file,),
                                 rounds=1,
                                 warmup_rounds=0,
                                 iterations=1)

    slvr = mdl_out.solvers[0]

    delta_qoi = slvr.Qval_ts[-1] - slvr.Qval_ts[0]
    u_norm = norm(slvr.U)

    pytest.check_float_result(delta_qoi, expected_delta_qoi)
    pytest.check_float_result(u_norm, expected_u_norm)

    # with open(pytest.temp_results, 'a') as output:
    #     output.write(f"{toml_file} - delta - {delta}\n")
    #     output.write(f"{toml_file} - u_norm - {u_norm}\n")

    # existing_temp_model["expected_delta_qoi"] = delta_qoi
    # existing_temp_model["expected_u_norm"] = u_norm


@pytest.mark.dependency()
@pytest.mark.runs
@pytest.mark.benchmark()
def test_run_eigendec(existing_temp_model, monkeypatch, benchmark, setup_deps, request):

    setup_deps.set_case_dependency(request, ["test_run_inv"])

    work_dir = existing_temp_model["work_dir"]
    toml_file = existing_temp_model["toml_filename"]

    # Switch to the working directory
    monkeypatch.chdir(work_dir)

    # Get expected values from the toml file
    params = config.ConfigParser(toml_file, top_dir=work_dir)
    expected_evals_sum = params.testing.expected_evals_sum
    expected_evec0_norm = params.testing.expected_evec0_norm

    EQReset()

    mdl_out = benchmark.pedantic(run_eigendec.run_eigendec,
                                 args=(toml_file, ),
                                 rounds=1,
                                 warmup_rounds=0,
                                 iterations=1)

    slvr = mdl_out.solvers[0]

    evals_sum = np.sum(slvr.eigenvals)
    evec0_norm = norm(slvr.eigenfuncs[0])

    # with open(pytest.temp_results, 'a') as output:
    #     output.write(f"{toml_file} - evals_sum - {evals_sum}\n")
    #     output.write(f"{toml_file} - evec0_norm - {evec0_norm}\n")

    # existing_temp_model["expected_evals_sum"] = evals_sum
    # existing_temp_model["expected_evec0_norm"] = evec0_norm

    pytest.check_float_result(evals_sum, expected_evals_sum)
    pytest.check_float_result(evec0_norm, expected_evec0_norm)

@pytest.mark.dependency()
@pytest.mark.runs
@pytest.mark.benchmark()
def test_run_errorprop(existing_temp_model, monkeypatch, benchmark, setup_deps, request):

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

    mdl_out = benchmark.pedantic(run_errorprop.run_errorprop,
                                 args=(toml_file, ),
                                 rounds=1,
                                 warmup_rounds=0,
                                 iterations=1)


    Q_sigma = mdl_out.Q_sigma[-1]
    Q_sigma_prior = mdl_out.Q_sigma_prior[-1]

    # existing_temp_model["expected_Q_sigma"] = Q_sigma
    # existing_temp_model["expected_Q_sigma_prior"] = Q_sigma_prior

    pytest.check_float_result(Q_sigma, expected_Q_sigma)
    pytest.check_float_result(Q_sigma_prior, expected_Q_sigma_prior)

@pytest.mark.dependency()
@pytest.mark.runs
@pytest.mark.benchmark()
def test_run_invsigma(existing_temp_model, monkeypatch, benchmark, setup_deps, request):

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

    mdl_out = benchmark.pedantic(run_invsigma.run_invsigma,
                                 args=(toml_file, ),
                                 rounds=1,
                                 warmup_rounds=0,
                                 iterations=1)

    cntrl_sigma_norm = norm(mdl_out.cntrl_sigma)
    cntrl_sigma_prior_norm = norm(mdl_out.cntrl_sigma_prior)

    # with open(pytest.temp_results, 'a') as output:
    #     output.write(f"{toml_file} - cntrl_sigma - {mdl_out.cntrl_sigma}\n")
    #     output.write(f"{toml_file} - cntrl_sigma_prior - {mdl_out.cntrl_sigma_prior}\n")

    # existing_temp_model["expected_cntrl_sigma_norm"] = cntrl_sigma_norm
    # existing_temp_model["expected_cntrl_sigma_prior_norm"] = cntrl_sigma_prior_norm

    pytest.check_float_result(cntrl_sigma_norm, expected_cntrl_sigma_norm)
    pytest.check_float_result(cntrl_sigma_prior_norm, expected_cntrl_sigma_prior_norm)

def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """

    if False:
        with open("pickled_values.p", 'wb') as pickle_out:
            pickle.dump(pytest.active_cases, pickle_out)
