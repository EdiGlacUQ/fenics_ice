# -*- coding: utf-8 -*-

import pytest
import pytest_mpi
from pathlib import Path
import shutil
from mpi4py import MPI
from logging import getLogger

import fenics_ice as fice
from aux import gen_rect_mesh #, gen_ismipC_domain, Uobs_from_momsolve  # TODO - refactor

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "short: tests which run quickly"
    )
    config.addinivalue_line(
        "markers", "long: tests which run slowly"
    )


@pytest.fixture
def repo_dir():
    """Get the git repo directory (to get example_cases, etc)"""
    return Path(fice.__file__).parents[1]

@pytest.fixture
def data_dir(repo_dir):
    """Get the directory containing test data"""
    return Path(repo_dir/"tests"/"data")

@pytest.fixture
def case_gen(repo_dir, counter):
    """Yield cases from example_cases"""
    case_dir = repo_dir/"example_cases"
    cases = [x for x in case_dir]
    return cases[counter]

@pytest.fixture
def one_case(repo_dir):
    """Yield cases from example_cases"""
    case_dir = repo_dir/"example_cases"
    if MPI.COMM_WORLD.size > 1:
        return case_dir / "ismipc_rc_1e6_parallel"  # no periodic_bc
    else:
        return case_dir / "ismipc_rc_1e6"

@pytest.fixture
def ismipc_temp_model(mpi_tmpdir, data_dir, one_case):
    """
    Set up an ismip test case from HDF5 datafiles

    Returns the path of the toml file
    """

    tmpdir = mpi_tmpdir

    # Find the toml file for ismipc case
    toml_files = [f for f in one_case.glob("ismip*toml")]
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

        # Copy the config toml to tmpdir
        shutil.copy(one_case/toml_file, tmpdir)

        # Generate ismip meshes & data
        # TODO - ought this to be stored in an HDF5 too?
        L = 40000
        gen_rect_mesh.main(10, 10, 0, L, 0, L,
                           str(destdir/"ismip_mesh.xml"))

    comm.barrier()
    return Path(tmpdir), toml_file.name
