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

#!/usr/bin/env python

from fenics_ice.backend import Function, Vector, function_get_values

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import resource

from pathlib import Path
import datetime

# assure we're not using tlm_adjoint version
from fenics_ice.eigendecomposition import eigendecompose
from fenics_ice.eigendecomposition import PythonMatrix, slepc_monitor_callback, slepc_config_callback
import fenics_ice.eigendecomposition as ED

from fenics_ice import model, solver, prior, inout

from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
from fenics_ice.decorators import count_calls, timer, flagged_error

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def run_eigendec(config_file):
    """
    Run the eigendecomposition phase of the model.

    1. Define the model domain & fields
    2. Runs the forward model w/ alpha/beta from run_inv
    3. Computes the Hessian of the *misfit* cost functional (J)
    4. Performs the generalized eigendecomposition with
        A = H_mis, B = prior_action
    """
    # Read run config file
    params = ConfigParser(config_file)
    log = inout.setup_logging(params)
    inout.log_preamble("eigendecomp", params)

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    # Get mesh & define model
    mesh = fice_mesh.get_mesh(params)
    mdl = model.model(mesh, input_data, params)
    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()
    mdl.bglen_from_data(mask_only=True)

    # Setup our solver object
    slvr = solver.ssa_solver(mdl, mixed_space=params.inversion.dual)

    cntrl = slvr.get_control()[0]
    space = slvr.get_control_space()

    # Regularization operator using inversion delta/gamma values
    Prior = mdl.get_prior()
    reg_op = Prior(slvr, space)

    msft_flag = params.eigendec.misfit_only
    if msft_flag:
        slvr.zero_inv_params()

    # Hessian Action
    slvr.set_hessian_action(cntrl)

    # Mass matrix solver
    xg, xb = Function(space), Function(space)

    # test, trial = TestFunction(space), TrialFunction(space)
    # mass = assemble(inner(test, trial) * slvr.dx)
    # mass_solver = KrylovSolver("cg", "sor")
    # mass_solver.parameters.update({"absolute_tolerance": 1.0e-32,
    #                                "relative_tolerance": 1.0e-14})
    # mass_solver.set_operator(mass)

    # Uncomment to get low-level SLEPc/PETSc output
    # set_log_level(10)

    @count_calls()
    # @timer
    def ghep_action(x):
        """Hessian action w/o preconditioning"""
        _, _, ddJ_val = slvr.ddJ.action(cntrl, x)
        # reg_op.inv_action(ddJ_val.vector(), xg.vector()) <- gnhep_prior
        return function_get_values(ddJ_val)

    @count_calls()
    def prior_action(x):
        """Define the action of the B matrix (prior)"""
        reg_op.action(x.vector(), xg.vector())
        return function_get_values(xg)

    # opts = {'prior': gnhep_prior_action, 'mass': gnhep_mass_action}
    # gnhep_func = opts[params.eigendec.precondition_by]

    num_eig = params.eigendec.num_eig
    n_iter = params.eigendec.power_iter  # <- not used yet

    # Hessian eigendecomposition using SLEPSc
    eig_algo = params.eigendec.eig_algo
    if eig_algo == "slepc":

        assert not flagged_error[0]

        results = {}  # Create this empty dict & pass it to slepc_monitor_callback to fill
        # Eigendecomposition
        import slepc4py.SLEPc as SLEPc
        esolver = eigendecompose(space,
                                 ghep_action,
                                 tolerance=params.eigendec.tol,
                                 max_it=params.eigendec.max_iter,
                                 N_eigenvalues=num_eig,
                                 problem_type=SLEPc.EPS.ProblemType.GHEP,
                                 solver_type=SLEPc.EPS.Type.KRYLOVSCHUR,
                                 configure=slepc_config_callback(reg_op, prior_action, space),
                                 monitor=slepc_monitor_callback(params, space, results))

        log.info("Finished eigendecomposition")
        vr = results['vr']
        lam = results['lam']

        if flagged_error[0]:
            # Note: I have been unable to confirm that this does anything in my setup
            # Python errors within LaplacianPC seem to be raised even without the
            # @flag_errors decorator.
            raise Exception("Python errors in eigendecomposition preconditioner.")

        # Check the eigenvectors & eigenvalues
        if(params.eigendec.test_ed):
            ED.test_eigendecomposition(esolver, results, space, params)

            if num_eig > 100:
                log.warning("Requesting inner product of more than 100 EVs, this is expensive!")
            # Check for B (not B') orthogonality & normalisation
            for i in range(num_eig):
                reg_op.action(vr[i].vector(), xg.vector())
                norm = xg.vector().inner(Vector(vr[i].vector())) ** 0.5
                if (abs(1.0 - norm) > params.eigendec.tol):
                    raise Exception(f"Eigenvector norm is {norm}")

            for i in range(num_eig):
                reg_op.action(vr[i].vector(), xg.vector())
                for j in range(i+1, num_eig):
                    inn = xg.vector().inner(Vector(vr[j].vector()))
                    if(abs(inn) > params.eigendec.tol):
                        raise Exception(f"Eigenvectors {i} & {j} inner product nonzero: {inn}")

        # Uses extreme amounts of disk space; suitable for ismipc only
        # #Save eigenfunctions
        # vtkfile = File(os.path.join(outdir,'vr.pvd'))
        # for v in vr:
        #     v.rename('v', v.label())
        #     vtkfile << v
        #
        # vtkfile = File(os.path.join(outdir,'vi.pvd'))
        # for v in vi:
        #     v.rename('v', v.label())
        #     vtkfile << v

    else:
        raise NotImplementedError

    slvr.eigenvals = lam
    slvr.eigenfuncs = vr

    # Plot of eigenvals
    lpos = np.argwhere(lam > 0)
    lneg = np.argwhere(lam < 0)
    lind = np.arange(0, len(lam))
    plt.semilogy(lind[lpos], lam[lpos], '.')
    plt.semilogy(lind[lneg], np.abs(lam[lneg]), '.')
    plt.savefig(os.path.join(params.io.output_dir, 'lambda.pdf'))
    plt.close()

    # Note - for now this does nothing, but eventually if the whole series
    # of runs were done without re-initializing solver, it'd be important to
    # put the inversion params back
    if msft_flag:
        slvr.set_inv_params()

    return mdl

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_eigendec(sys.argv[1])

    mem_high_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory high water mark: %s kb" % mem_high_kb)  # TODO - log, and put in a module
