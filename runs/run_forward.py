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

import sys
import os
from pathlib import Path
import argparse
from fenics import *
from tlm_adjoint_fenics import *

from fenics_ice import model, solver, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
import fenics_ice.fenics_util as fu

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from IPython import embed

stop_annotating()

def run_forward(config_file):

    # Read run config file
    params = ConfigParser(config_file)
    log = inout.setup_logging(params)
    inout.log_preamble("forward", params)

    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model
    mdl = model.model(mesh, input_data, params)

    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()

    # Solve
    slvr = solver.ssa_solver(mdl)
    slvr.save_ts_zero()

    cntrl = slvr.get_control()

    qoi_func = slvr.get_qoi_func()

    # TODO here - cntrl now returns a list - so compute_gradient returns a list of tuples

    # Run the forward model
    Q = slvr.timestep(adjoint_flag=1, qoi_func=qoi_func)
    # Run the adjoint model, computing gradient of Qoi w.r.t cntrl
    dQ_ts = compute_gradient(Q, cntrl)  # Isaac 27


    ## Temporary taylor verification

    object.__setattr__(slvr.params.time, "num_sens", 1)  # 1 qoi value only

    slvr.reset_ts_zero()
    J = slvr.timestep(adjoint_flag=1, qoi_func=qoi_func)[0]
    dJ = compute_gradient(J, cntrl)

    def forward_ts(cntrl, cntrl_init, name):
        slvr.reset_ts_zero()
        if(name == 'alpha'):
            slvr.alpha = cntrl
        elif(name == 'beta'):
            slvr.beta = cntrl
        else:
            raise ValueError(f"Unrecognised cntrl name: {name}")

        result = slvr.timestep(adjoint_flag=1, qoi_func=slvr.get_qoi_func())[0]

        # Reset after simulation - confirmed necessary
        if(name == 'alpha'):
            slvr.alpha = cntrl_init
        elif(name == 'beta'):
            slvr.beta = cntrl_init
        else:
            raise ValueError(f"Bad control name {name}")

        return result

    cntrl_init = [f.copy(deepcopy=True) for f in cntrl]
    #seeds = {'alpha': 1e-2} <- works for the ismipc case!

    seeds = {'alpha': 1e-2, 'beta': 1e-1}

    for cntrl_curr, cntrl_curr_init, dJ_curr in zip(cntrl, cntrl_init, dJ):

        min_order = taylor_test(lambda cntrl_val: forward_ts(cntrl_val,
                                                             cntrl_curr_init,
                                                             cntrl_curr.name()),
                                cntrl_curr,
                                J_val=J.value(),
                                dJ=dJ_curr,
                                seed=seeds[cntrl_curr.name()],
                                M0=cntrl_curr_init,
                                size=6)
        print(f"Forward simulation cntrl: {cntrl_curr.name()} min_order: {min_order}")
        # assert(min_order > 1.99)

    # Output model variables in ParaView+Fenics friendly format
    # Output QOI & DQOI (needed for next steps)
    inout.write_qval(slvr.Qval_ts, params)
    inout.write_dqval(dQ_ts, [var.name() for var in cntrl], params)

    # Output final velocity, surface & thickness (visualisation)
    inout.write_variable(slvr.U, params, name="U_fwd")
    inout.write_variable(mdl.surf, params, name="surf_fwd")

    H = project(mdl.H, mdl.Q)
    inout.write_variable(H, params, name="H_fwd")

    return mdl


if __name__ == "__main__":
    stop_annotating()

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_forward(sys.argv[1])
