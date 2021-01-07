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

import pytest
from pytest_dependency import depends
import re
from pathlib import Path
import shutil
import toml
import pickle

import fenics_ice as fice
import fenics_ice.fenics_util as fu
from aux import gen_rect_mesh

# Global variables
pytest.repo_dir = Path(fice.__file__).parents[1]
pytest.data_dir = Path(pytest.repo_dir/"tests"/"data")
pytest.case_dir = pytest.repo_dir/"example_cases"
pytest.active_cases = []

# Define the cases & their mesh characteristics (meshes generated on the fly)
pytest.case_list = []
pytest.case_list.append({"case_dir": "ismipc_rc_1e6",
                         "toml_file": "ismipc_rc_1e6.toml",
                         "serial": True,
                         "mesh_nx": 20,
                         "mesh_ny": 20,
                         "mesh_L": 40000})

pytest.case_list.append({"case_dir": "ismipc_rc_1e4",
                         "toml_file": "ismipc_rc_1e4.toml",
                         "serial": True,
                         "mesh_nx": 20,
                         "mesh_ny": 20,
                         "mesh_L": 40000})

pytest.case_list.append({"case_dir": "ismipc_30x30",
                         "toml_file": "ismipc_30x30.toml",
                         "serial": True,
                         "mesh_nx": 30,
                         "mesh_ny": 30,
                         "mesh_L": 40000})

pytest.case_list.append({"case_dir": "ice_stream",
                         "toml_file": "ice_stream.toml",
                         "serial": False,
                         "data_dir": pytest.case_dir / "ice_stream",
                         "tv_settings": {"obs": { "vel_file": "ice_stream_U_obs_tv.h5"},
                                         "constants": {"glen_n": 2.0},
                                         "ice_dynamics": {"allow_flotation": False}
                         }
})

def check_float_result(value, expected, work_dir, value_name, tol=None):
    """
    Compare scalar float against expected value.

    Also helps to document new expected values when cases/code changes.
    This functionality uses 'work_dir' to check which of
    pytest.active_cases is to be appended to. There's probably a better way
    to do this, though.
    """

    # MPI runs exhibit more variability (non-deterministic solvers?)
    if tol is None:
        if pytest.parallel:
            tol = 1e-8
        else:
            tol = 1e-9

    if not pytest.remake_cases:
        # Check against expected value
        rel_change = abs((value - expected) / value)
        assert rel_change < tol, f"Expected value: {expected} " \
            f"Computed value: {value} Tolerance: {tol}"
    else:
        # Store new 'expected' value rather than actually testing
        # TODO - is there a more robust way of checking this is the correct case?
        # Case dirs should be unique by design...
        for c in pytest.active_cases:
            if c['work_dir'] == work_dir:
                c[value_name] = value


pytest.check_float_result = check_float_result

def pytest_addoption(parser):
    """Option to run all (currently 3) test cases - or just 1 (default)"""
    parser.addoption("--all", action="store_true", help="run all combinations")
    parser.addoption("--remake", action="store_true",
                     help="Store new 'expectd values' to file instead of testing")

def pytest_configure(config):

    from mpi4py import MPI
    pytest.parallel = MPI.COMM_WORLD.size > 1

    config.addinivalue_line(
        "markers", "short: tests which run quickly"
    )
    config.addinivalue_line(
        "markers", "runs: tests of whole run components"
    )
    config.addinivalue_line(
        "markers", "tv: taylor verification"
    )

    pytest.remake_cases = config.option.remake


class DependencyGetter:
    """
    Handle parametrized test dependencies

    Given parametrized tests labelled_thus[0], set dependency
    to equivalent_previous[0] test.
    """

    @staticmethod
    def get_request_id(request):
        """Extract the test ID suffix e.g. [1]"""
        return int(re.search(r"\[([0-9]+)\]", request.node.name).group(1))

    @staticmethod
    def set_case_dependency(request, func_list):
        """Actually set the dependency"""
        assert isinstance(func_list, list)
        test_id = DependencyGetter.get_request_id(request)
        deps = [(f+"[%d]") % test_id for f in func_list]
        depends(request, deps)

@pytest.fixture
def setup_deps():
    """Fixture to return the above function globally"""
    return DependencyGetter

def pytest_collection_modifyitems(config, items):
    """
    Add a 'skip' marker to any Taylor verification (tv) tests
    if we haven't requested it. In other words, only run tv tests
    if specifically requested.
    """
    keywordexpr = config.option.keyword
    markexpr = config.option.markexpr
    if keywordexpr or markexpr:
        return  # let pytest handle this

    skip_mymarker = pytest.mark.skip(reason='Taylor verification not selected')
    for item in items:
        if 'tv' in item.keywords:
            item.add_marker(skip_mymarker)

@pytest.fixture
def case_gen(request):
    """
    Yield cases from example_cases

    Pytest iterates the 'request' argument and the relevant dictionary
    is returned, specifying a case directory and info about the mesh.
    """
    return pytest.case_list[request.param]

def pytest_generate_tests(metafunc):
    """This iterates the 'request' argument to case_gen above"""

    # To select which cases to run in parallel/serial
    if "case_gen" in metafunc.fixturenames:

        if pytest.parallel:
            case_ids = [i for i, case in enumerate(pytest.case_list) if not case["serial"]]
        else:
            case_ids = [i for i, case in enumerate(pytest.case_list) if case["serial"]]

        if not metafunc.config.getoption("all"):
            case_ids = [case_ids[0]]

        metafunc.parametrize("case_gen", case_ids, indirect=True)


@pytest.fixture
def temp_model(request, mpi_tmpdir, case_gen):
    """Return a temporary copy of one of the test cases"""
    return create_temp_model(request, mpi_tmpdir, case_gen)

@pytest.fixture
def persistent_temp_model(request, mpi_tmpdir, case_gen):
    """Return a reusable copy of a test case for testing multiple run phases"""
    return create_temp_model(request, mpi_tmpdir, case_gen, persist=True)

def create_temp_model(request, mpi_tmpdir, case_gen, persist=False):
    """
    Set up an ismip test case from HDF5 datafiles

    Returns the path of the toml file
    """
    from mpi4py import MPI

    tv = request.node.get_closest_marker("tv") is not None

    tmpdir = mpi_tmpdir
    data_dir = pytest.data_dir

    case_dir = pytest.case_dir / case_gen['case_dir']

    # Find the toml file for case
    toml_file = case_dir / case_gen['toml_file']

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:

        # Test case expects to find data in 'input'
        destdir = tmpdir/"input"
        destdir.mkdir()

        # Load toml config (for modification)
        config = toml.load(toml_file)

        # Override settings with Taylor verification specific stuff
        if tv and "tv_settings" in case_gen:
            fu.dict_update(config, case_gen['tv_settings'])

        # Turn off inversion verbose to keep test output clean
        config['inversion']['verbose'] = False

        # If doing Taylor verification, only take 1 sample:
        config['time']['num_sens'] = 1

        # and write out the toml to tmpdir
        with open(tmpdir/toml_file.name, 'w') as toml_out:
            toml.dump(config, toml_out)

        # ##### File Copies #######

        # Get the directory of the input files
        indata_dir = pytest.data_dir
        if('data_dir' in case_gen):
            indata_dir = case_gen['data_dir'] / config['io']['input_dir']

        # Collect data files to be copied...
        copy_set = set()
        for f in config['io']:
            if "data_file" in f:
                copy_set.add(indata_dir/config['io'][f])

        # ...including velocity observations
        copy_set.add(indata_dir/config['obs']['vel_file'])

        # And copy them
        for f in copy_set:
            shutil.copy(f, destdir)

        # Copy or generate the mesh
        mesh_filename = config["mesh"]["mesh_filename"]
        mesh_file = (case_dir / "input" / mesh_filename)

        try:
            mesh_ff_filename = config["mesh"]["bc_filename"]
            mesh_ff_file = (case_dir / "input" / mesh_ff_filename)
        except KeyError:
            mesh_ff_file = None

        # Generate mesh if it doesn't exist
        # TODO - not totally happy w/ logic here:
        # ismipc tests generate their own meshes, ice_stream doesn't
        if not (destdir/mesh_filename).exists():

            if(mesh_file.exists()):
                shutil.copy(mesh_file, destdir)
                if mesh_file.suffix == ".xdmf":
                    shutil.copy(mesh_file.with_suffix(".h5"), destdir)

                if mesh_ff_file:
                    shutil.copy(mesh_ff_file, destdir)
                    shutil.copy(mesh_ff_file.with_suffix(".h5"), destdir)

            else:
                gen_rect_mesh.gen_rect_mesh(case_gen['mesh_nx'],
                                            case_gen['mesh_ny'],
                                            0, case_gen['mesh_L'],
                                            0, case_gen['mesh_L'],
                                            str(destdir/mesh_filename))

    case_gen["work_dir"] = Path(tmpdir)
    case_gen["toml_filename"] = toml_file.name

    if persist:
        pytest.active_cases.append(case_gen)

    comm.barrier()
    return case_gen

@pytest.fixture
def existing_temp_model(case_gen):
    ncases = len(pytest.active_cases)
    for i in range(ncases):
        if pytest.active_cases[i]['case_dir'] == case_gen['case_dir']:
            return pytest.active_cases[i]

def update_expected_values():
    """
    Use regex to update expected values in example_cases toml files

    This *could* be achieved using toml_load and toml_dump, but this would
    lose all formatting & comments (not ideal)
    """

    float_str = r"([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"

    for case in pytest.active_cases:
        toml_file = pytest.case_dir/case['case_dir']/case['toml_filename']
        new_toml_file = toml_file.parent / (toml_file.name + ".new")
        new_toml_output = open(new_toml_file, 'w')

        with open(toml_file, 'r') as toml_input:

            # Construct list of 'expected_' value regexes & replacements
            expected_list = []
            for k in case.keys():
                if 'expected_' in k:
                    var_name = k
                    var_value = case[k]
                    var_re = re.compile(var_name + " *= *"+float_str)
                    var_str = var_name + f" = {var_value}"  # TODO - specify formatting?
                    expected_list.append((var_re, var_str))

            for line in toml_input:

                # Cycle all expected values, searching line for match
                for exp_re, exp_repl in expected_list:
                    search = exp_re.search(line)
                    if search is not None:
                        line = exp_re.sub(exp_repl, line)
                        break

                new_toml_output.write(line)

        #Replace the old toml file
        new_toml_output.close()
        new_toml_file.replace(toml_file)

def pytest_sessionfinish(session, exitstatus):
    """Write out expected values if requested"""

    from mpi4py import MPI
    root = (MPI.COMM_WORLD.rank == 0)

    if pytest.remake_cases and root:
        with open("new_expected_solution_values.p", 'wb') as pickle_out:
            pickle.dump(pytest.active_cases, pickle_out)

        update_expected_values()
