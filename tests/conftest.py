# -*- coding: utf-8 -*-

import pytest
import pytest_mpi
from pytest_dependency import depends
import re
from pathlib import Path
import shutil
from mpi4py import MPI
import toml
import pickle

import fenics_ice as fice
from aux import gen_rect_mesh

# Global variables
pytest.repo_dir = Path(fice.__file__).parents[1]
pytest.data_dir = Path(pytest.repo_dir/"tests"/"data")
pytest.case_dir = pytest.repo_dir/"example_cases"
pytest.active_cases = []

# Define the cases & their mesh characteristics (meshes generated on the fly)
pytest.case_list = []
pytest.case_list.append({"case_dir": "ismipc_rc_1e6",
                         "mesh_nx": 20,
                         "mesh_ny": 20,
                         "mesh_L": 40000,
                         "mesh_filename": "ismip_mesh.xml"})

pytest.case_list.append({"case_dir": "ismipc_rc_1e4",
                         "mesh_nx": 20,
                         "mesh_ny": 20,
                         "mesh_L": 40000,
                         "mesh_filename": "ismip_mesh.xml"})

pytest.case_list.append({"case_dir": "ismipc_30x30",
                         "mesh_nx": 30,
                         "mesh_ny": 30,
                         "mesh_L": 40000,
                         "mesh_filename": "ismip_mesh.xml"})

def check_float_result(value, expected, work_dir, value_name, tol=1e-9):
    """
    Compare scalar float against expected value.

    Also helps to document new expected values when cases/code changes.
    This functionality uses 'work_dir' to check which of
    pytest.active_cases is to be appended to. There's probably a better way
    to do this, though.
    """

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
    config.addinivalue_line(
        "markers", "short: tests which run quickly"
    )
    config.addinivalue_line(
        "markers", "long: tests which run slowly"
    )
    config.addinivalue_line(
        "markers", "runs: tests of whole run components"
    )

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
    if "case_gen" in metafunc.fixturenames:
        num_cases = len(pytest.case_list)
        if metafunc.config.getoption("all"):
            end = num_cases
        else:
            end = 1
        metafunc.parametrize("case_gen", list(range(end)), indirect=True)

    pytest.remake_cases = metafunc.config.option.remake

@pytest.fixture
def temp_model(mpi_tmpdir, case_gen):
    """Return a temporary copy of one of the test cases"""
    return create_temp_model(mpi_tmpdir, case_gen)

@pytest.fixture
def persistent_temp_model(mpi_tmpdir, case_gen):
    """Return a reusable copy of a test case for testing multiple run phases"""
    return create_temp_model(mpi_tmpdir, case_gen, persist=True)

def create_temp_model(mpi_tmpdir, case_gen, persist=False):
    """
    Set up an ismip test case from HDF5 datafiles

    Returns the path of the toml file
    """
    tmpdir = mpi_tmpdir
    data_dir = pytest.data_dir

    case_dir = pytest.case_dir / case_gen['case_dir']

    # Find the toml file for ismipc case
    toml_files = [f for f in case_dir.glob("ismip*toml")]
    assert len(toml_files) == 1
    toml_file = toml_files[0]

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:

        # Test case expects to find data in 'input'
        destdir = tmpdir/"input"
        destdir.mkdir()

        # Copy the data files to tmpdir
        indata_name = "ismipc_input.h5"
        veldata_name = "ismipc_U_obs.h5"

        shutil.copy(data_dir/indata_name, destdir)
        shutil.copy(data_dir/veldata_name, destdir)

        # Bit of a hack - turn off inversion verbose
        # to keep test output clean
        config = toml.load(toml_file)
        config['inversion']['verbose'] = False
        # And write out the toml to tmpdir
        with open(tmpdir/toml_file.name, 'w') as toml_out:
            toml.dump(config, toml_out)

        # Generate mesh if it doesn't exist
        if not (destdir/case_gen["mesh_filename"]).exists():
            gen_rect_mesh.gen_rect_mesh(case_gen['mesh_nx'],
                               case_gen['mesh_ny'],
                               0, case_gen['mesh_L'],
                               0, case_gen['mesh_L'],
                               str(destdir/case_gen["mesh_filename"]))

    comm.barrier()

    case_gen["work_dir"] = Path(tmpdir)
    case_gen["toml_filename"] = toml_file.name

    if persist:
        pytest.active_cases.append(case_gen)

    return case_gen

@pytest.fixture
def existing_temp_model(case_gen):
    ncases = len(pytest.active_cases)
    for i in range(ncases):
        if pytest.active_cases[i]['case_dir'] == case_gen['case_dir']:
            return pytest.active_cases[i]

def pytest_sessionfinish(session, exitstatus):
    """Write out expected values if requested"""

    if pytest.remake_cases:
        with open("new_expected_solution_values.p", 'wb') as pickle_out:
            pickle.dump(pytest.active_cases, pickle_out)
