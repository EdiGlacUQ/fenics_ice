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
# along with fenics_ice.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python
# =========================
# LICENSE: SLEPc for Python
# =========================
#
# :Author:  Lisandro Dalcin
# :Contact: dalcinl@gmail.com
#
#
# Copyright (c) 2015, Lisandro Dalcin.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from tlm_adjoint.fenics import function_get_values, function_global_size, \
    function_local_size, function_set_values, is_function, space_comm, \
    space_new

from fenics_ice import prior
from fenics import norm, project

import pickle
import numpy as np
import petsc4py.PETSc as PETSc
from fenics import HDF5File, XDMFFile
from pathlib import Path
import os
import logging

log = logging.getLogger("fenics_ice")

__all__ = \
    [
        "EigendecompositionException",
        "eigendecompose"
    ]


class EigendecompositionException(Exception):
    pass


_flagged_error = [False]


def flag_errors(fn):
    def wrapped_fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except:  # noqa: E722
            _flagged_error[0] = True
            raise
    return wrapped_fn


class PythonMatrix:
    def __init__(self, action, space):
        self._action = action
        self._space = space

    @flag_errors
    def mult(self, A, x, y):
        X = space_new(self._space)
        x_a = x.getArray(readonly=True)
        function_set_values(X, x_a)
        y_a = self._action(X)
        if is_function(y_a):
            y_a = function_get_values(y_a)
        if not np.can_cast(y_a, PETSc.ScalarType):
            raise EigendecompositionException("Invalid dtype")
        if y_a.shape != (y.getLocalSize(),):
            raise EigendecompositionException("Invalid shape")
        y.setArray(y_a)


def eigendecompose(space, A_action, B_matrix=None, N_eigenvalues=None,
                   solver_type=None, problem_type=None, which=None,
                   tolerance=1.0e-12, max_it=1e6, configure=None, monitor=None):
    # First written 2018-03-01
    """
    Matrix-free interface with SLEPc via slepc4py, loosely following
    the slepc4py 3.6.0 demo demo/ex3.py, for use in the calculation of Hessian
    eigendecompositions.

    Arguments:

    space          Eigenspace.
    A_action       Function handle accepting a function and returning a
                   function or NumPy array, defining the action of the
                   left-hand-side matrix, e.g. as returned by
                   Hessian.action_fn.
    B_matrix       (Optional) Right-hand-side matrix in a generalized
                   eigendecomposition.
    N_eigenvalues  (Optional) Number of eigenvalues to attempt to find.
                   Defaults to a full eigendecomposition.
    solver_type    (Optional) The solver type.
    problem_type   (Optional) The problem type. If not supplied
                   slepc4py.SLEPc.EPS.ProblemType.NHEP or
                   slepc4py.SLEPc.EPS.ProblemType.GNHEP are used.
    which          (Optional) Which eigenvalues to find. Defaults to
                   slepc4py.SLEPc.EPS.Which.LARGEST_MAGNITUDE.
    tolerance      (Optional) Tolerance, using slepc4py.SLEPc.EPS.Conv.REL
                   convergence criterion.
    configure      (Optional) Function handle accepting the EPS. Can be used
                   for manual configuration.
    monitor        (Optional) Function handle accepting the EPS. Can be used
                   for monitoring/outputting intermediate EVs.

    Returns:

    A tuple (lam, V_r) for Hermitian problems, or (lam, (V_r, V_i)) otherwise,
    where lam is an array of eigenvalues, and V_r / V_i are tuples of functions
    containing the real and imaginary parts of the corresponding eigenvectors.
    """

    import slepc4py.SLEPc as SLEPc

    if problem_type is None:
        if B_matrix is None:
            problem_type = SLEPc.EPS.ProblemType.NHEP
        else:
            problem_type = SLEPc.EPS.ProblemType.GNHEP
    if which is None:
        which = SLEPc.EPS.Which.LARGEST_MAGNITUDE

    X = space_new(space)
    n, N = function_local_size(X), function_global_size(X)
    del X
    N_ev = N if N_eigenvalues is None else N_eigenvalues

    comm = space_comm(space)  # .Dup()

    A_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                        PythonMatrix(A_action, space),
                                        comm=comm)
    A_matrix.setUp()

    esolver = SLEPc.EPS().create(comm=comm)
    if solver_type is not None:
        esolver.setType(solver_type)
    esolver.setProblemType(problem_type)
    if B_matrix is None:
        esolver.setOperators(A_matrix)
    else:
        esolver.setOperators(A_matrix, B_matrix)
    esolver.setWhichEigenpairs(which)
    esolver.setDimensions(nev=N_ev,
                          ncv=SLEPc.DECIDE, mpd=SLEPc.DECIDE)
    esolver.setConvergenceTest(SLEPc.EPS.Conv.REL)
    esolver.setTolerances(tol=tolerance, max_it=max_it)
    if configure is not None:
        configure(esolver)
    esolver.setUp()

    assert not _flagged_error[0]

    if monitor is not None:
        esolver.setMonitor(monitor)

    esolver.solve()
    if _flagged_error[0]:
        raise EigendecompositionException("Error encountered in "
                                          "SLEPc.EPS.solve")
    if esolver.getConverged() < N_ev:
        raise EigendecompositionException("Not all requested eigenpairs "
                                          "converged")

    return esolver

def test_eigendecomposition(esolver, results, space, params):
    """Check the consistency of the eigendecomposition"""

    # How many EVs?
    num_eig = params.eigendec.num_eig
    N = function_global_size(space_new(space))
    N_ev = N if num_eig is None else num_eig

    A_matrix, _ = esolver.getOperators()

    assert esolver.isHermitian(), "Expected Hermitian problem"

    V_r = results['vr']
    lam = results['lam']

    for i, (V_r, lami) in enumerate(zip(V_r, lam)):
        # Check it's an eigenvector
        residual = ev_resid(esolver, V_r, lami)
        log.info(f"Residual norm for eigenvector {i} is {residual}")

    # Check uniqueness of eigenvalues
    esolver_tol = esolver.getTolerances()[0]
    lam_diff = (lam - np.roll(lam, -1))[:-1]
    if not np.all(lam_diff > (esolver_tol**2.0)):
        log.warning("Eigenvalues are not unique!")

def slepc_config_callback(reg_op, prior_action, space):
    """Closure to define the slepc config callback"""
    def inner_fn(config):
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

        config.setOperators(A_matrix, B_matrix)
        config.view()

    return inner_fn

def slepc_monitor_callback(params, space, result_list):
    """
    Closure which defines the slepc monitor callback

    This allows keeping and modifying non-local variables, params etc
    """
    nconv_prev = 0

    num_eig = params.eigendec.num_eig if params.eigendec.num_eig is not None \
        else function_global_size(space_new(space))

    # dual = params.inversion.dual
    # alpha_active = params.inversion.alpha_active
    # beta_active = params.inversion.beta_active

    # Setup result dictionary
    result_list["lam"] = np.full(num_eig, np.NAN, dtype=np.float64)
    result_list["vr"] = []

    # Open results files
    ev_filepath = Path(params.io.output_dir) / params.io.eigenvecs_file
    # Delete files to avoid append
    ev_filepath.unlink(missing_ok=True)

    p = ev_filepath
    ev_xdmf_filepath = Path(p).parent / Path(p.stem + "_vis").with_suffix(".xdmf")

    ev_xdmf_file = XDMFFile(space.mesh().mpi_comm(), str(ev_xdmf_filepath))
    ev_xdmf_file.parameters["rewrite_function_mesh"] = False
    ev_xdmf_file.parameters["functions_share_mesh"] = True
    ev_xdmf_file.parameters["flush_output"] = True

    # Open and close files to ensure they are clear for future appends
    ev_file = HDF5File(space.mesh().mpi_comm(), str(ev_filepath), 'w')
    ev_file.close()

    lam_file = Path(params.io.output_dir) / params.io.eigenvalue_file

    V_r_prev = None

    def inner_fn(eps, its, nconv, eig, err):
        """A monitor callback for SLEPc.EPS to provide incremental output"""
        nonlocal nconv_prev
        nonlocal space
        nonlocal V_r_prev

        log.info(f"{nconv} EVs converged at iteration {its}")

        A_matrix, _ = eps.getOperators()

        ev_file = HDF5File(space.mesh().mpi_comm(), str(ev_filepath), 'a')

        for i in range(nconv_prev, min(nconv, num_eig)):
            V_r = space_new(space)
            v_r = A_matrix.getVecRight()
            lam_i = eps.getEigenpair(i, v_r)

            result_list["lam"][i] = lam_i.real
            function_set_values(V_r, v_r.getArray())
            V_r.rename("ev", "")
            result_list["vr"].append(V_r)
            ev_file.write(V_r, 'v', i)
            ev_xdmf_file.write(V_r, i)
            # for v, name in zip((V_r.sub(0), V_r.sub(1)), ['va', 'vb']):
            #     # ev_xdmf_file.write_checkpoint(v, name, i, append=True)
            #     v.rename(name, '')
            #     ev_xdmf_file.write(v, i)

        # Note: here we rewrite this pickle file every time, but given
        # the small amount of data, that's probably OK.
        pfile = open(lam_file, "wb")
        pickle.dump([result_list["lam"],
                     params.eigendec.num_eig,
                     params.eigendec.power_iter,
                     params.eigendec.eig_algo,
                     params.eigendec.misfit_only,
                     params.io.output_dir,
                     params.io.input_dir], pfile)
        pfile.close()

        ev_file.parameters.add("num_eig", nconv)
        # ev_file.parameters.add("eig_algo", eig_algo)
        # ev_file.parameters.add("timestamp", str(datetime.datetime.now()))

        ev_file.close()
        # ev_xdmf_file.close()
        nconv_prev = nconv

        test_ed = params.eigendec.test_ed
        if(test_ed):
            if its > 1:
                for i, (vr, vr_prev) in enumerate(zip(result_list["vr"], V_r_prev)):
                    diff_norm = norm(project(vr - vr_prev, space))
                    log.info(f"Norm diff between iterations for ev {i} is {diff_norm}")

            V_r_prev = [vr.copy(deepcopy=True) for vr in result_list["vr"]]

        log.info("Done with monitor")

    return inner_fn

def ev_resid(esolver, V_r, lam):
    """Given a function V_r, what is the residual norm, i.e. norm(A V_r - lambda B V_r)"""
    A, B = esolver.getOperators()

    x = A.getVecRight()
    y = B.getVecRight()
    A.mult(V_r.vector().vec(), x)  # confirmed this is ghep_action(vr)
    B.mult(V_r.vector().vec(), y)  # confirmed this is prior_action(vr)

    A_term = x.array
    B_term = y.array * lam
    resid = A_term - B_term
    resid_norm = np.linalg.norm(resid)

    return resid_norm
