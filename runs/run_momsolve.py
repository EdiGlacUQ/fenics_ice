"""
Run the momentum solver only. Primarily used to generate velocity field
for ismip-c case before running the main model.
"""

import sys
import os
from pathlib import Path

from fenics import *

from fenics_ice import model, solver, inout
import fenics_ice.mesh as fice_mesh
from fenics_ice.config import ConfigParser

from IPython import embed


def run_momsolve(config_file):

    # Read run config file
    params = ConfigParser(config_file)

    log = inout.setup_logging(params)

    input_data = inout.InputData(params)

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)
    # Initialize model
    mdl = model.model(mesh, input_data, params, init_vel_obs=False)
    # Get alpha from file
    mdl.alpha_from_data()

    try:
        Bglen = mdl.input_data.interpolate("Bglen", mdl.M)
        mdl.init_beta(mdl.bglen_to_beta(Bglen), False)
    except (AttributeError, KeyError) as e:
        log.warning('Using default bglen (constant)')

    # Forward Solve
    slvr = solver.ssa_solver(mdl)
    slvr.def_mom_eq()
    slvr.solve_mom_eq()


    # Output model variables in ParaView+Fenics friendly format
    outdir = params.io.output_dir

    h5file = HDF5File(mesh.mpi_comm(), str(Path(outdir)/'U.h5'), 'w')
    h5file.write(slvr.U, 'U')
    h5file.write(mesh, 'mesh')
    h5file.attributes('mesh')['periodic'] = params.mesh.periodic_bc

    inout.write_variable(slvr.U, params)



if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_momsolve(sys.argv[1])
