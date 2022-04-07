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

# -*- coding: utf-8 -*-

from fenics_ice.backend import Function, function_update_state, norm

import pytest
import os
import numpy as np
import fenics_ice as fice
from fenics_ice import model, config, inout, solver

def init_model(model_dir, toml_file):

    # Switch to the working directory
    os.chdir(model_dir)
    assert (model_dir/toml_file).exists()

    # Get the relevant filenames from test case
    params = config.ConfigParser(toml_file, model_dir)
    dd = params.io.input_dir
    data_file = params.io.data_file

    # Read the input data & mesh
    indata = inout.InputData(params)
    inmesh = fice.mesh.get_mesh(params)

    # Create the model object
    return model.model(inmesh, indata, params,
                       init_fields=False, init_vel_obs=False)

def initialize_fields(mdl):
    """Initialize data fields in model object"""
    mdl.init_fields_from_data()

    # Add random noise to Beta field iff we're inverting for it
    mdl.bglen_from_data()
    mdl.init_beta(mdl.bglen_to_beta(mdl.bglen),
                  mdl.params.inversion.beta_active)

def initialize_vel_obs(mdl):
    """Initialize velocity observations"""
    mdl.vel_obs_from_data()
    mdl.init_lat_dirichletbc()

### Tests below this point ###

@pytest.mark.dependency()
def test_init_model(temp_model):
    """Check the initialization of the model"""
    work_dir = temp_model["work_dir"]
    toml_file = temp_model["toml_filename"]

    mdl = init_model(work_dir, toml_file)

    assert mdl.mesh is not None
    assert mdl.input_data is not None

    assert mdl.Q is not None
    assert mdl.Qp is not None
    assert mdl.V is not None
    assert mdl.M is not None

    return mdl

@pytest.mark.dependency()
def test_initialize_fields(request, setup_deps, temp_model):
    """Attempt to initialize fields from HDF5 data file"""

    # testcount = get_request_id(request)
    # depends(request, ["test_init_model[%d]" % testcount])
    setup_deps.set_case_dependency(request, ["test_init_model"])
#    set_case_dependency(request,
    work_dir = temp_model["work_dir"]
    toml_file = temp_model["toml_filename"]
    mdl = init_model(work_dir, toml_file)
    initialize_fields(mdl)

    # Size of vectors in Q function space
    Q_size = mdl.Q.tabulate_dof_coordinates().shape[0]
    M_size = mdl.M.tabulate_dof_coordinates().shape[0]

    assert mdl.bed.vector()[:].size == Q_size
    assert mdl.surf.vector()[:].size == Q_size

    return mdl


@pytest.mark.dependency()
def test_initialize_vel_obs(request, setup_deps, temp_model):
    """Attempt to velocity observations from HDF5 data file"""
    setup_deps.set_case_dependency(request, ["test_initialize_fields"])

    work_dir = temp_model["work_dir"]
    toml_file = temp_model["toml_filename"]
    mdl = init_model(work_dir, toml_file)
    initialize_vel_obs(mdl)

    assert mdl.u_obs_Q is not None  # TODO - better test here
    assert mdl.vel_obs['uv_obs_pts'].size > 0
    assert np.linalg.norm(mdl.latbc.vector()[:]) != 0.0


@pytest.mark.dependency()
def test_gen_init_alpha(request, setup_deps, temp_model):
    """Attempt to generate initial guess for alpha"""
    setup_deps.set_case_dependency(request, ["test_init_model",
                                             "test_initialize_fields"])

    work_dir = temp_model["work_dir"]
    toml_file = temp_model["toml_filename"]

    mdl = init_model(work_dir, toml_file)
    initialize_fields(mdl)
    initialize_vel_obs(mdl)

    expected_init_alpha = mdl.params.testing.expected_init_alpha

    # Generate initial guess for alpha
    mdl.gen_alpha()

    alpha_norm = norm(mdl.alpha.vector())
    # TODO - won't properly set expected value when --remake, because
    # pytest.active_cases doesn't exist yet
    pytest.check_float_result(alpha_norm,
                              expected_init_alpha,
                              work_dir, 'expected_init_alpha')

@pytest.mark.dependency()
def test_writers(request, setup_deps, temp_model):
    """Test the Writers in inout"""
    setup_deps.set_case_dependency(request, ["test_init_model"])

    work_dir = temp_model["work_dir"]
    toml_file = temp_model["toml_filename"]

    mdl = init_model(work_dir, toml_file)

    # Create test function for writing
    space = mdl.Q
    test_fun = Function(space, name="test")
    test_fun.vector()[:] = -1.0

    vtkpath = inout.gen_path(mdl.params, "test", ".pvd")
    xdmfpath = vtkpath.with_suffix(".xdmf")

    xdmfWriter = inout.XDMFWriter(xdmfpath, comm=mdl.mesh.mpi_comm())
    vtkWriter = inout.VTKWriter(vtkpath, comm=mdl.mesh.mpi_comm())

    # Check can write to file without error
    vtkWriter.write(test_fun, step=1)
    xdmfWriter.write(test_fun, step=1)

    # Check file is produced
    assert vtkpath.exists()
    assert xdmfpath.exists()

    # Check can't write unstepped variable to stepped file
    with pytest.raises(ValueError):
        vtkWriter.write(test_fun)

    with pytest.raises(ValueError):
        xdmfWriter.write(test_fun)

    # New writers for unstepped output

    xdmfWriter = inout.XDMFWriter(xdmfpath)
    vtkWriter = inout.VTKWriter(vtkpath)

    # Can't write unstepped to XDMF
    xdmfWriter.write(test_fun)
    vtkWriter.write(test_fun)

    # Check fail from attempting stepped output
    with pytest.raises(ValueError):
        vtkWriter.write(test_fun, step=1)

    with pytest.raises(ValueError):
        xdmfWriter.write(test_fun, step=1)

@pytest.mark.dependency()
def test_control_separation(request, setup_deps, temp_model):
    """Check that mdl.alpha and slvr.alpha are distinct functions"""

    setup_deps.set_case_dependency(request, ["test_init_model",
                                             "test_initialize_fields"])
    work_dir = temp_model["work_dir"]
    toml_file = temp_model["toml_filename"]

    mdl = init_model(work_dir, toml_file)
    initialize_fields(mdl)
    initialize_vel_obs(mdl)
    slvr = solver.ssa_solver(mdl)

    slvr.alpha.vector()[:] = 1e3
    slvr.alpha.vector().apply("insert")
    function_update_state(slvr.alpha)
    slvr.beta.vector()[:] = 1e3
    slvr.beta.vector().apply("insert")
    function_update_state(slvr.beta)

    with pytest.raises(AttributeError):
        slvr.beta = 1.0
    with pytest.raises(AttributeError):
        slvr.alpha = 1.0

    norm_as = norm(slvr.alpha)
    norm_bs = norm(slvr.beta)
    norm_am = norm(mdl.alpha)
    norm_bm = norm(mdl.beta)

    assert norm_as != norm_am
    assert norm_bs != norm_bm

# Unused!
def override_param(param_section, name, value):
    """Override frozen ConfigParser params for testing"""
    try:
        param_section.__getattribute__(name)
    except AttributeError:
        raise

    object.__setattr__(param_section, name, value)
