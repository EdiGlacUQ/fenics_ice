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

# assure we're not using tlm_adjoint version
from fenics_ice.eigendecomposition import eigendecompose as fice_ed
from fenics_ice.eigendecomposition import PythonMatrix
from fenics_ice import model, solver, prior, inout

from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
from fenics_ice.decorators import count_calls, timer, flagged_error

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

        (n, N), (n_col, N_col) = A_matrix.getSizes()
        assert n == n_col
        assert N == N_col
        del n_col, N_col

        comm = A_matrix.getComm()

        B_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                            PythonMatrix(prior_action, space),
                                            comm=comm)
        B_matrix.setUp()

        config.view()  # TODO - should this go to log?
        config.setOperators(A_matrix, B_matrix)

    nconv_prev = 0
    def slepc_monitor_callback(eps, its, nconv, eig, err):
        """A monitor callback for SLEPc.EPS to provide incremental output"""
        nonlocal nconv_prev
        log.info(f"{nconv} EVs converged at iteration {its}")

        A_matrix, _ = eps.getOperators()
        v_r = A_matrix.getVecRight()

        for i in range(nconv_prev, nconv):
            lam = eps.getEigenpair(i, v_r)
            log.warning(f"monitor ev{i} norm: {v_r.norm()}")

        nconv_prev = nconv


    # opts = {'prior': gnhep_prior_action, 'mass': gnhep_mass_action}
    # gnhep_func = opts[params.eigendec.precondition_by]

    num_eig = params.eigendec.num_eig
    n_iter = params.eigendec.power_iter  # <- not used yet

    # Hessian eigendecomposition using SLEPSc
    eig_algo = params.eigendec.eig_algo
    if eig_algo == "slepc":

        assert not flagged_error[0]

        # Eigendecomposition
        lam, vr = fice_ed(space,
                          ghep_action,
                          tolerance=1.0e-10,
                          N_eigenvalues=num_eig,
                          problem_type=SLEPc.EPS.ProblemType.GHEP,
                          # solver_type=SLEPc.EPS.Type.ARNOLDI,
                          configure=slepc_config_callback,
                          monitor=slepc_monitor_callback)

        if flagged_error[0]:
            # Note: I have been unable to confirm that this does anything in my setup
            # Python errors within LaplacianPC seem to be raised even without the
            # @flag_errors decorator.
            raise Exception("Python errors in eigendecomposition preconditioner.")

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

        p = Path(ev_file)
        ev_xdmf = str(p.parent / Path(p.stem + "_vis").with_suffix(".xdmf"))
        with XDMFFile(slvr.mesh.mpi_comm(),
                      ev_xdmf) as xdmf_ev:
            for i, v in enumerate(vr):
                v.rename("ev", "")
                xdmf_ev.write(v, i)

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
