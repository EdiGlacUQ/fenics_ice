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
import sys
import resource

import os
from fenics import *
from tlm_adjoint_fenics import *
import pickle
from pathlib import Path
import datetime

from fenics_ice import model, solver, prior, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
from fenics_ice.decorators import count_calls, flag_errors, timer

import slepc4py.SLEPc as SLEPc
import petsc4py.PETSc as PETSc

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

    dd = params.io.input_dir
    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model
    mdl = model.model(mesh, input_data, params)

    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()

    # Setup our solver object
    slvr = solver.ssa_solver(mdl)

    # TODO generalise - get_control returns a list
    cntrl = slvr.get_control()[0]
    space = cntrl.function_space()

    msft_flag = params.eigendec.misfit_only
    if msft_flag:
        slvr.zero_inv_params()

    # Hessian Action
    slvr.set_hessian_action(cntrl)
    slvr.set_GN_action(cntrl)

    # Mass matrix solver
    xg, xb = Function(space), Function(space)

    # test, trial = TestFunction(space), TrialFunction(space)
    # mass = assemble(inner(test, trial) * slvr.dx)
    # mass_solver = KrylovSolver("cg", "sor")
    # mass_solver.parameters.update({"absolute_tolerance": 1.0e-32,
    #                                "relative_tolerance": 1.0e-14})
    # mass_solver.set_operator(mass)

    # Regularization operator using inversion delta/gamma values
    # TODO - this won't handle dual inversion case
    if params.inversion.alpha_active:
        delta = params.inversion.delta_alpha
        gamma = params.inversion.gamma_alpha
    elif params.inversion.beta_active:
        delta = params.inversion.delta_beta
        gamma = params.inversion.gamma_beta

    reg_op = prior.laplacian(delta, gamma, space)

    # Uncomment to get low-level SLEPc/PETSc output
    # set_log_level(10)

    @count_calls()
    # @timer
    def ghep_action(x):
        """Hessian action w/o preconditioning"""
        _, _, ddJ_val = slvr.ddJ.action(cntrl, x)
        # reg_op.inv_action(ddJ_val.vector(), xg.vector()) <- gnhep_prior
        return function_get_values(ddJ_val)

    def ghep_GN_action(x):
        """Hessian action w/o preconditioning"""
        ddJ_val = slvr.H_GN.action(cntrl, x)
        # reg_op.inv_action(ddJ_val.vector(), xg.vector()) <- gnhep_prior
        return function_get_values(ddJ_val)

    @count_calls()
    def prior_action(x):
        """Define the action of the B matrix (prior)"""
        reg_op.action(x.vector(), xg.vector())
        return function_get_values(xg)

    def prior_approx_action(x):
        """Only used for checking B' orthonormality"""
        reg_op.approx_action(x.vector(), xg.vector())
        return function_get_values(xg)

    class PythonMatrix:
        """
        Define a 'shell' matrix defined only by its
        action (mult) on a vector (x)
        Copied from tlm_adjoint eigendecomposition.py
        """

        def __init__(self, action, X):
            self._action = action
            self._X = X

        @flag_errors
        def mult(self, A, x, y):
            function_set_values(self._X, x.getArray(readonly=True))
            y.setArray(self._action(self._X))

    def slepc_config_callback(config):
        log.info("Got to the callback")

        # KSP corresponds to B-matrix inversion
        # Set it to precondition only because we
        # supply the inverse in LaplacianPC
        ksp = config.getST().getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(prior.LaplacianPC(reg_op))

        # A_matrix already defined so just grab it
        A_matrix, _ = config.getOperators()

        # Equivalent code to tlm_adjoint for defining the shell matrix
        Y = space_new(space)
        n, N = function_local_size(Y), function_global_size(Y)
        B_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                            PythonMatrix(prior_action, Y),
                                            comm=function_comm(Y))
        B_matrix.setUp()

        config.view()  # TODO - should this go to log?
        config.setOperators(A_matrix, B_matrix)

    # opts = {'prior': gnhep_prior_action, 'mass': gnhep_mass_action}
    # gnhep_func = opts[params.eigendec.precondition_by]

    num_eig = params.eigendec.num_eig
    n_iter = params.eigendec.power_iter  # <- not used yet

    # Hessian eigendecomposition using SLEPSc
    eig_algo = params.eigendec.eig_algo
    if eig_algo == "slepc":

        # Eigendecomposition
        lam, vr = eigendecompose(space,
                                 ghep_GN_action,
                                 tolerance=1.0e-10,
                                 N_eigenvalues=num_eig,
                                 problem_type=SLEPc.EPS.ProblemType.GHEP,
                                 # solver_type=SLEPc.EPS.Type.ARNOLDI,
                                 configure=slepc_config_callback)

        # Check orthonormality of EVs
        if num_eig is not None and num_eig < 100:

            # Check for B (not B') orthogonality & normalisation
            for i in range(num_eig):
                reg_op.action(vr[i].vector(), xg.vector())
                norm = xg.vector().inner(Vector(vr[i].vector())) ** 0.5
                print("EV %s norm %s" % (i, norm))

            for i in range(num_eig):
                reg_op.action(vr[i].vector(), xg.vector())
                for j in range(i+1, num_eig):
                    inn = xg.vector().inner(Vector(vr[j].vector()))
                    print("EV %s %s inner %s" % (i, j, inn))

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

        ev_file = params.io.eigenvecs_file
        with HDF5File(slvr.mesh.mpi_comm(),
                      os.path.join(outdir, ev_file), 'w') as hdf5file:
            for i, v in enumerate(vr):
                hdf5file.write(v, 'v', i)

            hdf5file.parameters.add("num_eig", num_eig)
            hdf5file.parameters.add("eig_algo", eig_algo)
            hdf5file.parameters.add("timestamp", str(datetime.datetime.now()))


    else:
        raise NotImplementedError

    slvr.eigenvals = lam
    slvr.eigenfuncs = vr

    # Save eigenvals and some associated info - TODO HDF5File?
    fileout = params.io.eigenvalue_file
    pfile = open(os.path.join(outdir, fileout), "wb")
    pickle.dump([lam, num_eig, n_iter, eig_algo, msft_flag, outdir, dd], pfile)
    pfile.close()

    # Plot of eigenvals
    lamr = lam.real
    lpos = np.argwhere(lamr > 0)
    lneg = np.argwhere(lamr < 0)
    lind = np.arange(0, len(lamr))
    plt.semilogy(lind[lpos], lamr[lpos], '.')
    plt.semilogy(lind[lneg], np.abs(lamr[lneg]), '.')
    plt.savefig(os.path.join(outdir, 'lambda.pdf'))

    # Note - for now this does nothing, but eventually if the whole series
    # of runs were done without re-initializing solver, it'd be important to
    # put the inversion params back
    if msft_flag:
        slvr.set_inv_params()

    return mdl

if __name__ == "__main__":
    stop_annotating()

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_eigendec(sys.argv[1])

    mem_high_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory high water mark: %s kb" % mem_high_kb)  # TODO - log, and put in a module
