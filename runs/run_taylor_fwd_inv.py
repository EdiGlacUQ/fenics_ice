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

# This script does not output anything. It is only use for
# running taylor test verification on the forward runs outside
# of the pytest environment

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys

from runs import run_forward

from fenics_ice.backend import clear_caches, compute_gradient, \
    reset_manager, stop_manager, taylor_test

def EQReset():
    """Take care of tlm_adjoint EquationManager"""
    # This prevents checkpointing errors when these run phases
    # are tested after the stuff in test_model.py
    reset_manager("memory", {})
    clear_caches()
    stop_manager()

def test_tv_run_forward(config_file):

    EQReset()

    mdl_out = run_forward.run_forward(config_file)

    slvr = mdl_out.solvers[0]

    qoi_func = slvr.get_qoi_func()
    cntrl = slvr.get_control()

    slvr.reset_ts_zero()
    J = slvr.timestep(adjoint_flag=1, qoi_func=qoi_func)[2]
    dJ = compute_gradient(J, cntrl)

    def forward_ts(cntrl, cntrl_init, name):
        slvr.reset_ts_zero()
        if (name == 'alpha'):
            slvr.set_control_fns([cntrl, slvr._beta], initial=True)
        elif (name == 'beta'):
            slvr.set_control_fns([slvr._alpha, cntrl], initial=True)
        else:
            raise ValueError(f"Unrecognised cntrl name: {name}")

        result = slvr.timestep(adjoint_flag=1, qoi_func=slvr.get_qoi_func())[2]
        stop_manager()

        # Reset after simulation - confirmed necessary
        if (name == 'alpha'):
            slvr.set_control_fns([cntrl_init, slvr.beta], initial=True)
        elif (name == 'beta'):
            slvr.set_control_fns([slvr._alpha, cntrl_init], initial=True)
        else:
            raise ValueError(f"Bad control name {name}")

        return result

    cntrl_init = [f.copy(deepcopy=True) for f in cntrl]

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
        print(min_order)
        assert (min_order > 1.95)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    test_tv_run_forward(sys.argv[1])
