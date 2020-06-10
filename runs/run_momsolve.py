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

    dd = params.io.input_dir
    data_file = params.io.data_file
    input_data = inout.InputData(Path(dd) / data_file)

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    mdl = model.model(mesh, input_data, params)

    mdl.bed_from_data()
    mdl.thick_from_data()
    mdl.gen_surf()
    mdl.mask_from_data()
    mdl.bmelt_from_data()
    mdl.smb_from_data()
    mdl.alpha_from_data()
    mdl.label_domain()

    try:
        Bglen = mdl.input_data.interpolate("Bglen", mdl.M)
        mdl.init_beta(mdl.bglen_to_beta(Bglen), False)
    except AttributeError:
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

    vtkfile = File(os.path.join(outdir,'U.pvd'))
    vtkfile << slvr.U



if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_momsolve(sys.argv[1])
