# -*- coding: utf-8 -*-

import pytest
from fenics import *
from tlm_adjoint import *
import fenics_ice as fice
from fenics_ice import model, config, inout, solver

def test_init_model(ismipc_temp_model, monkeypatch):
    """Check the initialization of the model"""
    case_dir, toml_file = ismipc_temp_model

    # Switch to the working directory
    monkeypatch.chdir(case_dir)

    assert (case_dir/toml_file).exists()

    # Get the relevant filenames from test case
    params = config.ConfigParser(toml_file, case_dir)
    dd = params.io.input_dir
    data_file = params.io.data_file

    # Read the input data & mesh
    indata = inout.InputData(case_dir / dd / data_file)
    inmesh = fice.mesh.get_mesh(params)

    # Only checking initialization doesn't throw an error
    mdl = model.model(inmesh, indata, params)

    assert mdl.mesh is not None
    assert mdl.input_data is not None

    assert mdl.Q is not None
    assert mdl.Qp is not None
    assert mdl.V is not None
    assert mdl.M is not None

    return mdl


def test_initialize_fields(ismipc_temp_model, monkeypatch):
    """Attempt to initialize fields from HDF5 data file"""
    mdl = test_init_model(ismipc_temp_model, monkeypatch)

    mdl.bed_from_data()
    mdl.thick_from_data()
    mdl.gen_surf()
    mdl.mask_from_data()
    mdl.init_vel_obs()
    mdl.bmelt_from_data()
    mdl.smb_from_data()
    mdl.init_lat_dirichletbc()
    mdl.label_domain()

    mdl.gen_alpha()

    # Add random noise to Beta field iff we're inverting for it
    mdl.bglen_from_data()
    mdl.init_beta(mdl.bglen_to_beta(mdl.bglen),
                  mdl.params.inversion.beta_active)

    # Size of vectors in Q function space
    Q_size = mdl.Q.tabulate_dof_coordinates().shape[0]
    M_size = mdl.M.tabulate_dof_coordinates().shape[0]

    assert mdl.bed.vector()[:].size == Q_size
    assert mdl.surf.vector()[:].size == Q_size
    assert mdl.mask.vector()[:].size == M_size

    assert mdl.uv_obs_pts is not None

    return mdl

@pytest.mark.benchmark(min_rounds=1, max_time=1)  # <- just run it once
@pytest.mark.long
def test_solver_inversion(ismipc_temp_model, monkeypatch, benchmark):
    """Run solver.inversion() for benchmarking"""
    expected_J_inv = 181139.25789039143

    mdl = test_initialize_fields(ismipc_temp_model, monkeypatch)

    slvr = solver.ssa_solver(mdl)

    assert slvr.dt is not None
    assert slvr.dObs is not None
    assert slvr.dIce is not None
    assert slvr.dx is not None
    assert slvr.uv_obs_pts is mdl.uv_obs_pts

    # Set tolerances down a little for testing
    slvr.params.inversion.inv_options['ftol'] = 1e-2
    slvr.params.inversion.inv_options['gtol'] = 1e-8
    slvr.params.inversion.inv_options['disp'] = 0

    # Time the inversion
    benchmark(slvr.inversion)

    # Check the cost functional against expected value
    rel_change = abs(slvr.J_inv.value() - expected_J_inv) \
        / expected_J_inv
    assert rel_change < 1e-7

def override_param(param_section, name, value):
    """Override frozen ConfigParser params for testing"""

    try:
        param_section.__getattribute__(name)
    except AttributeError:
        raise

    object.__setattr__(param_section, name, value)

@pytest.mark.long
@pytest.mark.benchmark(min_rounds=1, max_time=1)
@pytest.mark.parametrize("adjoint_flag,save",
                         [(False, False), (True, False), (True, True)],
                         ids=["A-S-", "A+S-", "A+S+"])
def test_solver_timestep(ismipc_temp_model, monkeypatch, benchmark, save, adjoint_flag):
    """
    Run solver.timestep() for benchmarking

    This is not a realistic run because, to avoid reliance on other tests,
    no realistic value for alpha or beta is set.
    """

    # TODO - this will only be valid with ismipc_rc_1e6
    # (or 1e4, but not 30x30, 40x40)
    expected_delta_qoi = 1159538497.5
    expected_q_is = 1600001780418369.0

    # Only run 10 steps to save time
    tot_steps = 10
    num_sens = 5

    # Use the initialization test to get a model object
    mdl = test_initialize_fields(ismipc_temp_model, monkeypatch)

    # Set analytical alpha to avoid dependence on inversion
    mdl.alpha_from_data()

    # Override the number of steps defined in the TOML
    override_param(mdl.params.time, 'total_steps', tot_steps)
    override_param(mdl.params.time, 'num_sens', num_sens)

    slvr = solver.ssa_solver(mdl)
    slvr.save_ts_zero()

    qoi_func = slvr.get_qoi_func()

    q_is = benchmark(slvr.timestep,
                     save=save,
                     adjoint_flag=1,
                     qoi_func=qoi_func)

    assert len(slvr.Qval_ts) == tot_steps + 1
    delta = slvr.Qval_ts[-1] - slvr.Qval_ts[0]

    assert abs(delta - expected_delta_qoi) < 1.0
    assert abs(q_is[0].value() - expected_q_is) < 1.0
