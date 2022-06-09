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
# running taylor test verification on the inversion outside
# of the pytest environment

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from runs import run_inv
from fenics_ice.backend import clear_caches, reset_manager, stop_manager,\
    taylor_test_tlm, taylor_test_tlm_adjoint

def EQReset():
    """Take care of tlm_adjoint EquationManager"""
    # This prevents checkpointing errors when these run phases
    # are tested after the stuff in test_model.py
    reset_manager("memory", {})
    clear_caches()
    stop_manager()

def test_tv_run_inversion(config_file):

    EQReset()

    # Run the thing
    mdl_out = run_inv.run_inv(config_file)

    # Get expected values from the toml file
    alpha_active = mdl_out.params.inversion.alpha_active
    beta_active = mdl_out.params.inversion.beta_active

    if alpha_active:
        fwd_alpha = mdl_out.solvers[0].forward
        alpha = mdl_out.solvers[0].alpha

        min_order = taylor_test_tlm(fwd_alpha,
                                    alpha,
                                    tlm_order=1,
                                    seed=1.0e-5)

        assert (min_order > 1.95)

        min_order = taylor_test_tlm_adjoint(fwd_alpha,
                                            alpha,
                                            adjoint_order=1,
                                            seed=1.0e-5)

        assert (min_order > 1.95)

        min_order = taylor_test_tlm_adjoint(fwd_alpha,
                                            alpha,
                                            adjoint_order=2,
                                            seed=1.0e-5)
        assert (min_order > 1.95)

    if beta_active:
        fwd_beta = mdl_out.solvers[0].forward
        beta = mdl_out.solvers[0].beta

        min_order = taylor_test_tlm(fwd_beta,
                                    beta,
                                    tlm_order=1,
                                    seed=1.0e-5)
        assert (min_order > 1.95)

        min_order = taylor_test_tlm_adjoint(fwd_beta,
                                            beta,
                                            adjoint_order=1,
                                            seed=1.0e-5)
        assert (min_order > 1.95)

        min_order = taylor_test_tlm_adjoint(fwd_beta,
                                            beta,
                                            adjoint_order=2,
                                            seed=1.0e-5)
        assert (min_order > 1.95)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    test_tv_run_inversion(sys.argv[1])
