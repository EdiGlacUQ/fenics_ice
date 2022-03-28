#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import OptimizationException, clear_caches, \
    function_assign, function_axpy, function_comm, function_copy, \
    function_get_values, function_inner, function_is_cached, \
    function_is_checkpointed, function_is_static, function_linf_norm, \
    function_new, function_set_values, is_function, restore_manager, \
    set_manager
from tlm_adjoint import manager as _manager

from collections import deque
from collections.abc import Sequence
import logging
import numpy as np

__all__ = \
    [
        "H_approximation",

        "line_search",
        "line_search_rank0_scipy_line_search",
        "line_search_rank0_scipy_scalar_search_wolfe1",
        "line_search_rank0_scipy_scalar_search_wolfe2",

        "l_bfgs",
        "minimize_l_bfgs"
    ]


def functions_assign(X, Y):
    assert len(X) == len(Y)
    for x, y in zip(X, Y):
        function_assign(x, y)


def functions_axpy(Y, alpha, X):
    assert len(Y) == len(X)
    for y, x in zip(Y, X):
        function_axpy(y, alpha, x)


def functions_copy(X):
    return tuple(function_copy(x) for x in X)


def functions_inner(X, Y):
    assert len(X) == len(Y)
    inner = 0.0
    for x, y in zip(X, Y):
        inner += function_inner(x, y)
    return inner


def functions_new(X):
    return tuple(function_new(x) for x in X)


def wrapped_action(M):
    M_arg = M

    def M(*X):
        M_X = M_arg(*X)
        if is_function(M_X):
            M_X = (M_X,)
        if len(M_X) != len(X):
            raise OptimizationException("Incompatible shape")
        return M_X

    return M


class H_approximation:
    """
    L-BFGS approximate Hessian inverse.

    Constructor arguments:
        m          Keep the last m vector pairs
        skip_atol  Absolute tolerance in the update skip test
        skip_rtol  Relative tolerance in the update skip test
        M          A callable defining the step inner product. Must not modify
                   input data. Defaults to identity. If supplied then M_inv
                   must be supplied.
        M_inv      A callable defining the gradient change inner product. Must
                   not modify input data. Defaults to identity. If supplied
                   then M must be supplied.
    """

    def __init__(self, m,
                 skip_atol=0.0, skip_rtol=1.0e-12, M=None, M_inv=None):
        if skip_atol < 0.0:
            raise OptimizationException("skip_atol must be non-negative")
        if skip_rtol < 0.0:
            raise OptimizationException("skip_rtol must be non-negative")
        if (M is None and M_inv is not None) or (M is not None and M_inv is None):  # noqa: E501
            raise OptimizationException("Cannot supply only one of M and "
                                        "M_inv")

        if M is None:
            def M(*X):
                return X  # copy not required
        else:
            M = wrapped_action(M)

        if M_inv is None:
            def M_inv(*X):
                return X  # copy not required
        else:
            M_inv = wrapped_action(M_inv)

        self._m = m
        self._skip_atol = skip_atol
        self._skip_rtol = skip_rtol
        self._M = M
        self._M_inv = M_inv

        self._iterates = deque()

    def append(self, S, Y, remove=True):
        """
        Add a vector pair. Skipped if the inner product is small or
        non-positive.

        Update skip approach similar to that taken in SciPy
            Git master 0a7bc723d105288f4b728305733ed8cb3c8feeb5, June 20 2020
            scipy/optimize/lbfgsb_src/lbfgsb.f
            mainlb subroutine
        and in (3.9) of
            R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, "A limited memory
            algorithm for bound constrained optimization", SIAM Journal on
            Scientific Computing 16(5), 1190--1208, 1995
        Specifically, given a step s and gradient change y, only accepts the
        update if
            y^T s > max(skip_atol, skip_rtol sqrt(| s^T M s y^T M_inv y |))
        where the |.| is used to work around possible underflows.

        Arguments:
            S       Step
            Y       Gradient change
            remove  Whether to remove any excess vector pairs
        Returns:
            (S_inner_Y, S_Y_added, S_Y_removed)
        with
            S_inner_Y    y^T s
            S_Y_added    Whether the given vector pair was added
            S_Y_removed  A list of any removed vector pairs
        """

        if is_function(S):
            S = (S,)
        if is_function(Y):
            Y = (Y,)
        if len(S) != len(Y):
            raise OptimizationException("Incompatible shape")

        if self._skip_rtol == 0.0:
            skip_tol = self._skip_atol
        else:
            skip_tol = max(
                self._skip_atol,
                self._skip_rtol * np.sqrt(abs(functions_inner(S, self._M(*S))
                                              * functions_inner(self._M_inv(*Y), Y))))  # noqa: E501

        S_inner_Y = functions_inner(S, Y)
        if S_inner_Y > skip_tol:
            rho = 1.0 / S_inner_Y
            self._iterates.append((rho, functions_copy(S), functions_copy(Y)))

            if remove:
                S_Y_removed = self.remove()
            else:
                S_Y_removed = []

            return S_inner_Y, True, S_Y_removed
        else:
            return S_inner_Y, False, []

    def remove(self):
        """
        Remove any excess vector pairs, returning the removed pairs as a list
        """

        S_Y_removed = []
        while len(self._iterates) > self._m:
            S_Y_removed.append(self._iterates.popleft())
        return S_Y_removed

    def reset(self):
        """
        Remove all vector pairs, returning the removed pairs as a list
        """

        S_Y_removed = list(self._iterates)
        self._iterates.clear()
        return S_Y_removed

    def action(self, X, H_0=None, theta=1.0):
        """
        Compute the action of the approximate Hessian inverse.

        Implementation of L-BFGS approximate Hessian inverse action, as in
        Algorithm 7.4 of
            J. Nocedal and S. J. Wright, Numerical optimization, second
            edition, Springer, 2006
        with theta scaling, see equation (3.2) of
            R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, "A limited memory
            algorithm for bound constrained optimization", SIAM Journal on
            Scientific Computing 16(5), 1190--1208, 1995

        Arguments:
            X     Vector on which to compute the action
            H_0   A callable defining the action of the unscaled initial
                  approximate Hessian inverse. Must correspond to the action of
                  a symmetric positive definite matrix. Must not modify input
                  data. Identity used if not supplied.
        """

        if is_function(X):
            X = (X,)
        X = functions_copy(X)

        if H_0 is None:
            def H_0(*X):
                return X  # copy not required
        else:
            H_0 = wrapped_action(H_0)

        alphas = []
        for rho, S, Y in reversed(self._iterates):
            alpha = rho * functions_inner(S, X)
            functions_axpy(X, -alpha, Y)
            alphas.append(alpha)
        alphas.reverse()

        R = functions_copy(H_0(*X))
        if not np.all(theta == 1.0):
            if isinstance(theta, (int, np.integer, float, np.floating)):
                theta = [theta for r in R]
            assert len(R) == len(theta)
            for r, th in zip(R, theta):
                function_set_values(r, function_get_values(r) / th)

        assert len(self._iterates) == len(alphas)
        for (rho, S, Y), alpha in zip(self._iterates, alphas):
            beta = rho * functions_inner(R, Y)
            functions_axpy(R, alpha - beta, S)

        return R[0] if len(R) == 1 else R

    def inverse_update_decomposition(self, B_0=None):
        """
        Writing equation (7.24) of
            J. Nocedal and S. J. Wright, Numerical optimization, second
            edition, Springer, 2006
        as
            B_k = B_0 - F_k^T G_k^{-1} F_k
        with
            G_k = [[S_k^T B_0 S_k, L_k], [L_k^T, -D_k]]
            F_k = [[S_k^T B_0], [Y_k^T]]
        and with S_k, L_k, D_k, and Y_k as defined in Theorem 7.4 of
            J. Nocedal and S. J. Wright, Numerical optimization, second
            edition, Springer, 2006
        build and return
            (G, G_solve, F)
        with
            G        A NumPy array corresponding to G_k
            G_solve  A callable
                         def G_solve(b):
                     where b is a NumPy array, and returning a NumPy array x
                     corresponding to the solution of G_k x = b
            F        A list of tuples of Function objects corresponding to the
                     rows of F_k
        G_solve, which solves equations involving the G_k matrix, uses the
        decomposition in equations (2.25) and (2.26) of
            R. H. Byrd, J. Nocedal, and R. B. Schnabel, ``Representation of
            quasi-Newton matrices and their use in limited memory methods'',
            Mathematical Programming 63, 129--156, 1994

        Arguments:
            B_0   A callable defining the action of the initial approximate
                  Hessian. Must correspond to the action of a symmetric
                  positive definite matrix. Must not modify input data.
                  Identity used if not supplied.
        """

        from scipy.linalg import cholesky, solve_triangular

        if B_0 is None:
            def B_0(*X):
                return X  # copy not required
        else:
            B_0 = wrapped_action(B_0)

        m = len(self._iterates)
        if m == 0:
            def G_solve(b):
                assert b.shape == (0,)
                return np.zeros_like(b)
            return G_solve, []

        F = [None for i in range(2 * m)]
        for i, (rho_i, S_i, Y_i) in enumerate(self._iterates):
            F[i] = functions_copy(B_0(*S_i))
            F[m + i] = functions_copy(Y_i)

        L = np.zeros((m, m), dtype=np.float64)
        for i, (rho_i, S_i, Y_i) in enumerate(self._iterates):
            for j, (rho_j, S_j, Y_j) in enumerate(self._iterates):
                if i > j:
                    L[i, j] = functions_inner(S_i, Y_j)

        G = np.zeros((2 * m, 2 * m), dtype=np.float64)
        G[:m, m:] = L
        G[m:, :m] = L.T
        for i, (rho_i, S_i, Y_i) in enumerate(self._iterates):
            for j in range(i + 1):
                G[i, j] = functions_inner(S_i, F[j])
                if i > j:
                    G[j, i] = G[i, j]
            G[m + i, m + j] = -functions_inner(S_i, Y_i)

        D_inv = -1.0 / np.diag(G[m:, m:])
        sqrt_D_inv = np.sqrt(D_inv)

        J_J_T = G[:m, :m] + L @ np.diag(D_inv) @ L.T
        J = cholesky(J_J_T, lower=True)

        def G_solve(b):
            assert b.shape == (2 * m,)
            x = np.zeros_like(b)
            x[m:] = sqrt_D_inv * b[m:]
            x[:m] = solve_triangular(J,
                                     b[:m] + L @ (sqrt_D_inv * x[m:]),
                                     trans="N", lower=True)
            y = np.zeros_like(b)
            y[:m] = solve_triangular(J, x[:m], trans="T", lower=True)
            y[m:] = -sqrt_D_inv * (x[m:] - sqrt_D_inv * (L.T @ y[:m]))
            return y

        return G, G_solve, F

    def inverse_action(self, X, B_0=None, B_approx_decomp=None):
        """
        Compute the action of the approximate Hessian.

        Computed using the representation in Theorem 7.4 of
            J. Nocedal and S. J. Wright, Numerical optimization, second
            edition, Springer, 2006

        Arguments:
            X    A Function, or list or tuple of Function objects, defining the
                 vector on which to compute the action
            B_0  A callable defining the action of the initial approximate
                 Hessian. Must correspond to the action of a symmetric positive
                 definite matrix. Must not modify input data. Identity used if
                 not supplied.
            B_approx_decomp
                 As returned by self.inverse_update_decomposition
         """

        if is_function(X):
            X = (X,)

        if B_0 is None:
            def B_0(*X):
                return X  # copy not required
        else:
            B_0 = wrapped_action(B_0)

        m = len(self._iterates)
        if B_approx_decomp is None:
            G, G_solve, F = self.inverse_update_decomposition(B_0=B_0)
        else:
            G, G_solve, F = B_approx_decomp

        F_X = np.zeros(2 * m, dtype=np.float64)
        for i in range(2 * m):
            F_X[i] = functions_inner(X, F[i])

        G_inv_F_x = G_solve(F_X)

        R = functions_copy(B_0(*X))
        for i in range(2 * m):
            functions_axpy(R, -G_inv_F_x[i], F[i])

        return R[0] if len(R) == 1 else R

    def inverse_update_eigendecomposition(self, atol, rtol,
                                          B_0=None, M=None, M_inv=None,
                                          B_approx_decomp=None,
                                          M_equals_B_0_simplifications=False,
                                          normalize=True,
                                          comm=None):
        """
        Compute an eigendecomposition of the Hessian update
            B_k - B_0,
        in the generalized eigendecomposition
            (B_k - B_0) v_k = lambda_k M v_k,
        and return eigenpairs with largest magnitude eigenvalues, where
            | lambda_k | >= max( atol, rtol max_l | lambda_l | ).
        Should only be used to find eigenpairs with associated eigenvalues
        non-zero.

        Arguments:
            atol   Absolute tolerance used to select the largest magnitude
                   eigenvalues. Must be non-negative.
            rtol   Relative tolerance used to select the largest magnitude
                   eigenvalues. Must be non-negative.
            B_0    A callable defining the action of the initial approximate
                   Hessian. Must correspond to the action of a symmetric
                   positive definite matrix. Must not modify input data.
                   Identity used if not supplied. If supplied then M_inv must
                   be supplied.
            M      A callable defining the action of a symmetric positive
                   definite matrix M. Must not modify input data. Defaults to
                   B_0. If supplied then M_inv must be supplied.
            M_inv  A callable defining the action of a symmetric positive
                   definite matrix equal to the inverse of the matrix M. Must
                   not modify input data. Defaults to identity. If supplied
                   then B_0 or M must be supplied.
            B_approx_decomp
                   As returned by self.inverse_update_decomposition
            M_equals_B_0_simplifications
                   If True then apply simplifications which assume that M=B_0
            normalize
                   Whether the returned eigenvectors should be normalized to
                   have unit M-norm
            comm   MPI communicator
        Returns
            (lambda, w)
        where lambda is a NumPy array of eigenvalues, sorted in order from
        largest to smallest magnitude, and w is a list of corresponding
        M-orthogonomal eigenvectors. If normalize is True, then the
        eigenvectors are normalized to have unit M-norm.
        """

        from scipy.linalg import eig

        if atol < 0.0:
            raise OptimizationException("atol must be non-negative")
        if rtol < 0.0:
            raise OptimizationException("rtol must be non-negative")

        if (B_0 is None and M is None) and M_inv is not None:
            raise OptimizationException("If M_inv is supplied, then B_0 or M "
                                        "must be supplied")
        if (B_0 is not None or M is not None) and M_inv is None:
            raise OptimizationException("If B_0 or M are supplied, then M_inv "
                                        "must be supplied")
        if B_0 is None:
            def B_0(*X):
                return X  # copy not required
        else:
            B_0 = wrapped_action(B_0)

        if M is None:
            M = B_0
        else:
            M = wrapped_action(M)

        if M_inv is None:
            def M_inv(*X):
                return X  # copy not required
        else:
            M_inv = wrapped_action(M_inv)

        m = len(self._iterates)
        if m == 0:
            return np.array([], dtype=np.float64), []
        if B_approx_decomp is None:
            G, G_solve, F = self.inverse_update_decomposition(B_0=B_0)
        else:
            G, G_solve, F = B_approx_decomp

        F_M_inv = [None for i in range(2 * m)]
        F_M_inv_F_T = np.zeros((2 * m, 2 * m), dtype=np.float64)
        if M_equals_B_0_simplifications:
            for i, (rho_i, S_i, Y_i) in enumerate(self._iterates):
                F_M_inv[i] = functions_copy(S_i)
                F_M_inv[m + i] = functions_copy(M_inv(*Y_i))
            F_M_inv_F_T[:m, :m] = G[:m, :m]
            for i, (rho_i, S_i, Y_i) in enumerate(self._iterates):
                for j, (rho_j, S_j, Y_j) in enumerate(self._iterates):
                    if i > j:
                        F_M_inv_F_T[i, m + j] = G[i, m + j]
                    elif i == j:
                        F_M_inv_F_T[i, m + j] = -G[m + i, m + j]
                    else:
                        F_M_inv_F_T[i, m + j] = functions_inner(S_i, Y_j)

                    if i >= j:
                        F_M_inv_F_T[m + i, m + j] = functions_inner(F_M_inv[m + j], Y_i)  # noqa: E501
                        if i > j:
                            F_M_inv_F_T[m + j, m + i] = F_M_inv_F_T[m + i, m + j]  # noqa: E501
            F_M_inv_F_T[m:, :m] = F_M_inv_F_T[:m, m:].T
        else:
            for i in range(2 * m):
                F_M_inv[i] = functions_copy(M_inv(*F[i]))
            for i in range(2 * m):
                for j in range(i + 1):
                    F_M_inv_F_T[i, j] = functions_inner(F_M_inv[j], F[i])
                    if i > j:
                        F_M_inv_F_T[j, i] = F_M_inv_F_T[i, j]

        W_V = np.zeros((2 * m, 2 * m), dtype=np.float64)
        for i in range(2 * m):
            W_V[i, :] = -G_solve(F_M_inv_F_T[i, :])

        lam, v = eig(W_V)

        # Synchronization check -- check that all processes have the same
        # eigenvalues
        if comm is None:
            comm = function_comm(F[0][0])
        comm = comm.Dup()
        lam_0 = comm.bcast(lam, root=0)
        comm.Free()
        del comm
        assert abs(lam - lam_0).max() == 0.0

        # Sort, and discard complex components
        eigenpairs = sorted([(lam[i], v[:, i]) for i in range(lam.shape[0])],
                            key=lambda eigenpair: abs(eigenpair[0].real),
                            reverse=True)
        lam = np.array([eigenpair[0].real for eigenpair in eigenpairs],
                       dtype=np.float64)
        v = [(eigenpair[1] / eigenpair[1][np.argmax(abs(eigenpair[1]))]).real
             for eigenpair in eigenpairs]
        del eigenpairs

        N_eigenpairs = 0
        while N_eigenpairs < len(lam) and abs(lam[N_eigenpairs]) >= max(atol, rtol * abs(lam[0])):  # noqa: E501
            N_eigenpairs += 1
        if N_eigenpairs == 0:
            return np.array([], dtype=np.float64), []
        lam = lam[:N_eigenpairs]
        v = v[:N_eigenpairs]

        G_inv_v = [G_solve(v[i]) for i in range(N_eigenpairs)]

        assert len(F_M_inv) > 0
        w = [functions_new(F_M_inv[0]) for i in range(N_eigenpairs)]
        for i in range(N_eigenpairs):
            for j in range(2 * m):
                functions_axpy(w[i], -G_inv_v[i][j], F_M_inv[j])

        if normalize:
            # Modified Gram-Schmidt
            for i in range(N_eigenpairs):
                for j in range(i):
                    functions_axpy(w[i], -functions_inner(w[i], M(*w[j])), w[j])  # noqa: E501
                w_norm = np.sqrt(abs(functions_inner(w[i], M(*w[i]))))
                for w_ in w[i]:
                    function_set_values(w_, function_get_values(w_) / w_norm)
        else:
            # Modified Gram-Schmidt, excluding normalization
            w_norm_sq = np.full(N_eigenpairs, np.NAN, dtype=np.float64)
            for i in range(N_eigenpairs):
                if i > 0:
                    w_norm_sq[i - 1] = abs(functions_inner(w[i - 1], M(*w[i - 1])))  # noqa: E501
                for j in range(i):
                    functions_axpy(w[i], -functions_inner(w[i], M(*w[j])) / w_norm_sq[j], w[j])  # noqa: E501

        if len(w[0]) == 1:
            w = [w[i][0] for i in range(N_eigenpairs)]
        return lam, w


def line_search_rank0_scipy_line_search(
        F, Fp, c1, c2, old_F_val=None, old_Fp_val=None, **kwargs):

    def f(x):
        return F(x[0])

    def myfprime(x):
        return np.array([Fp(x[0])])

    if old_Fp_val is not None:
        old_Fp_val = np.array([old_Fp_val])

    from scipy.optimize import line_search
    alpha, fc, gc, new_fval, old_fval, new_slope = line_search(
        f, myfprime, xk=np.array([0.0]), pk=np.array([1.0]),
        gfk=old_Fp_val, old_fval=old_F_val, c1=c1, c2=c2,
        **kwargs)
    if new_slope is None:
        alpha = None
    if alpha is None:
        new_fval = None
    return alpha, new_fval


def line_search_rank0_scipy_scalar_search_wolfe1(
        F, Fp, c1, c2, old_F_val=None, old_Fp_val=None, **kwargs):
    from scipy.optimize.linesearch import scalar_search_wolfe1 as line_search
    alpha, phi, phi0 = line_search(
        F, Fp,
        phi0=old_F_val, derphi0=old_Fp_val, c1=c1, c2=c2,
        **kwargs)
    if alpha is None:
        phi = None
    return alpha, phi


def line_search_rank0_scipy_scalar_search_wolfe2(
        F, Fp, c1, c2, old_F_val=None, old_Fp_val=None, **kwargs):
    from scipy.optimize.linesearch import scalar_search_wolfe2 as line_search
    alpha_star, phi_star, phi0, derphi_star = line_search(
        F, Fp,
        phi0=old_F_val, derphi0=old_Fp_val, c1=c1, c2=c2,
        **kwargs)
    if derphi_star is None:
        alpha_star = None
    if alpha_star is None:
        phi_star = None
    return alpha_star, phi_star


def line_search(F, Fp, X, minus_P, c1=1.0e-4, c2=0.9,
                old_F_val=None, old_Fp_val=None,
                line_search_rank0=line_search_rank0_scipy_line_search,
                line_search_rank0_kwargs={},
                comm=None):
    Fp = wrapped_action(Fp)

    if is_function(X):
        X_rank1 = (X,)
    else:
        X_rank1 = X
    del X

    if is_function(minus_P):
        minus_P = (minus_P,)
    if len(minus_P) != len(X_rank1):
        raise OptimizationException("Incompatible shape")

    if comm is None:
        comm = function_comm(X_rank1[0])
    comm = comm.Dup()

    last_F = [None, None]

    def F_rank0(x):
        X_rank0 = x
        del x
        X = functions_copy(X_rank1)
        functions_axpy(X, -X_rank0, minus_P)
        last_F[0] = float(X_rank0)
        last_F[1] = F(*X)
        return last_F[1]

    last_Fp = [None, None, None]

    def Fp_rank0(x):
        X_rank0 = x
        del x
        X = functions_copy(X_rank1)
        functions_axpy(X, -X_rank0, minus_P)
        last_Fp[0] = float(X_rank0)
        last_Fp[1] = functions_copy(Fp(*X))
        last_Fp[2] = -functions_inner(minus_P, last_Fp[1])
        return last_Fp[2]

    if old_F_val is None:
        old_F_val = F_rank0(0.0)

    if old_Fp_val is None:
        old_Fp_val_rank0 = Fp_rank0(0.0)
    else:
        if is_function(old_Fp_val):
            old_Fp_val = (old_Fp_val,)
        if len(old_Fp_val) != len(X_rank1):
            raise OptimizationException("Incompatible shape")
        old_Fp_val_rank0 = -functions_inner(minus_P, old_Fp_val)
    del old_Fp_val

    if comm.rank == 0:
        def F_rank0_bcast(x):
            comm.bcast(("F_rank0", (x,)), root=0)
            return F_rank0(x)

        def Fp_rank0_bcast(x):
            comm.bcast(("Fp_rank0", (x,)), root=0)
            return Fp_rank0(x)

        alpha, new_F_val = line_search_rank0(
            F_rank0_bcast, Fp_rank0_bcast, c1, c2,
            old_F_val=old_F_val, old_Fp_val=old_Fp_val_rank0,
            **line_search_rank0_kwargs)
        comm.bcast(("return", (alpha, new_F_val)), root=0)
    else:
        while True:
            action, data = comm.bcast(None, root=0)
            if action == "F_rank0":
                X_rank0, = data
                F_rank0(X_rank0)
            elif action == "Fp_rank0":
                X_rank0, = data
                Fp_rank0(X_rank0)
            elif action == "return":
                alpha, new_F_val = data
                break
            else:
                raise OptimizationException(f"Unexpected action '{action:s}'")

    comm.Free()

    if alpha is None:
        return None, old_Fp_val_rank0, None, None, None
    else:
        if new_F_val is None:
            if last_F[0] is not None and last_F[0] == alpha:
                new_F_val = last_F[1]
            else:
                new_F_val = F_rank0(alpha)

        if last_Fp[0] is not None and last_Fp[0] == alpha:
            new_Fp_val_rank1 = last_Fp[1]
            new_Fp_val_rank0 = last_Fp[2]
        else:
            new_Fp_val_rank0 = Fp_rank0(alpha)
            assert last_Fp[0] == alpha
            new_Fp_val_rank1 = last_Fp[1]
            assert last_Fp[2] == new_Fp_val_rank0

        return (alpha, old_Fp_val_rank0, new_F_val,
                new_Fp_val_rank1[0] if len(new_Fp_val_rank1) == 1 else new_Fp_val_rank1,  # noqa: E501
                new_Fp_val_rank0)


def l_bfgs(F, Fp, X0, m, s_atol, g_atol, converged=None, max_its=1000,
           H_0=None, theta_scale=True, block_theta_scale=True, delta=1.0,
           skip_atol=0.0, skip_rtol=1.0e-12, M=None, M_inv=None,
           c1=1.0e-4, c2=0.9,
           old_F_val=None,
           line_search_rank0=line_search_rank0_scipy_line_search,
           line_search_rank0_kwargs={},
           comm=None):
    """
    Minimization using L-BFGS, following Algorithm 7.5 of
        J. Nocedal and S. J. Wright, Numerical optimization, second edition,
        Springer, 2006

    Theta scaling is similar to that in
        R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, "A limited memory algorithm
        for bound constrained optimization", SIAM Journal on Scientific
        Computing 16(5), 1190--1208, 1995
    but using y_k^T H_0 y_k in place of y_k^T y_k, and with a general H_0. On
    the first iteration, and when restarting due to line search failures, theta
    is set equal to
        theta = { sqrt(| g^T M_inv g |) / delta   if delta is not None
                { 1                               if delta is None
    where g is the (previous) gradient vector -- see 'Implementation' in
    section 6.1 of
        J. Nocedal and S. J. Wright, Numerical optimization, second edition,
        Springer, 2006

    Restart on line search failure similar to approach described in
        C. Zhu, R. H. Byrd, P. Lu, and J. Nocedal, "Algorithm 778: L-BFGS-B:
        Fortran subroutines for large-scale bound-constrained optimization",
        ACM Transactions on Mathematical Software 23(4), 550--560, 1997
    but using gradient-descent defined using the H_0 norm in place of I

    Arguments:
        F          A callable defining the functional
        Fp         A callable defining the functional gradient
        X0         Initial guess
        m          Keep the last m vector pairs
        s_atol     Step absolute tolerance. If None then the step norm
                   convergence test is disabled.
        g_atol     Gradient absolute tolerance. If None then the gradient
                   change norm convergence test is disabled.
        converged  A callable of the form
                       def converged(it, F_old, F_new, X_new, G_new, S, Y):
                   where X_new, G_new, S, and Y are a Function, or a list or
                   tuple of Function objects, and with
                       it      The iteration number, an integer
                       F_old   The old value of the functional
                       F_new   The new value of the functioanl
                       X_new   The new value of X, with F_new = F(X_new)
                       G_new   The new gradient, Fp(X_new)
                       S       The step
                       Y       The gradient change
                   and returning True if the problem has converged, and False
                   otherwise. X_new, G_new, S, and Y must not be modified.
        max_its    Maximum number of iterations
        H_0        A callable defining the action of the unscaled initial
                   approximate Hessian inverse. Must correspond to the action
                   of a symmetric positive definite matrix. Must not modify
                   input data. Identity used if not supplied. If supplied then
                   M must be supplied.
        theta_scale  Whether to apply theta scaling (see above).
        block_theta_scale  Whether to apply separate theta scaling to each
                   control function. Intended to be used with block-diagonal
                   H_0 (and M_inv if delta is not None).
        delta      Defines the initial theta scaling (see above). If delta is
                   None then no scaling is applied on the first iteration, or
                   when restarting due to line search failures.
        skip_atol  Skip absolute tolerance (see H_approximation.append)
        skip_rtol  Skip relative tolerance (see H_approximation.append)
        M          A callable defining the action of a symmetric positive
                   definite matrix, used to define the step norm. Must not
                   modify input data. Identity used if not supplied. If
                   supplied then H_0 or M_inv must be supplied.
        M_inv      A callable defining the action of a symmetric positive
                   definite matrix, used to define the gradient norm. Must not
                   modify input data. Defaults to H_0. If supplied then M must
                   be supplied.
        c1, c2     Parameters in the Wolfe conditions. See section 3.1
                   (where values are suggested) and (3.6) of
                     J. Nocedal and S. J. Wright, Numerical optimization,
                     second edition, Springer, 2006
        old_F_val  Value of F at the initial guess
        line_search_rank0         See below.
        line_search_rank0_kwargs  See below.
        comm       MPI communicator

    line_search_rank0 is a callable implementing a one dimensional line search
    algorithm, yielding a value of alpha_k such that the Wolfe conditions are
    satisfied as defined in (3.6) of
        J. Nocedal and S. J. Wright, Numerical optimization, second edition,
        Springer, 2006
    for the case x_k=[0] and p_k=[1]. This has interface:
        def line_search_rank0(
            F, Fp, c1, c2, old_F_val=None, old_Fp_val=None, **kwargs):
    with arguments:
        F           A callable, with a floating point input x, and returning
                    the value of the functional F(x)
        Fp          A callable, with a floating point input x, and returning
                    the value of the functional derivative F'(x)
        c1, c2      Parameters in the Wolfe conditions. See (3.6) of
                      J. Nocedal and S. J. Wright, Numerical optimization,
                      second edition, Springer, 2006
        old_F_val   Value of the functional at x = 0, F(x = 0)
        old_Fp_val  Value of the functional at x = 0, F'(x = 0)
    and with remaining keyword arguments given by line_search_rank0_kwargs.
    This returns
        (alpha_k, new_F_val)
    with:
        alpha_k    Resulting value of alpha_k, or None on failure
        new_F_val  Value of the functional at F(x = alpha), or None if not
                   available

    Returns:
        (X, its, conv, reason, F_calls, Fp_calls, H_approx)
    with:
        X         Result of the minimization
        its       Iterations taken
        conv      Whether converged
        reason    A string describing the reason for return
        F_calls   Number of functional evaluation calls
        Fp_calls  Number of functional gradient evaluation calls
        H_approx  The inverse Hessian approximation
    """

    logger = logging.getLogger("fenics_ice.l_bfgs")

    F_arg = F
    F_calls = [0]

    def F(*X):
        F_calls[0] += 1
        return F_arg(*X)

    Fp_arg = Fp
    Fp_calls = [0]

    def Fp(*X):
        Fp_calls[0] += 1
        Fp_val = Fp_arg(*X)
        if is_function(Fp_val):
            Fp_val = (Fp_val,)
        if len(Fp_val) != len(X):
            raise OptimizationException("Incompatible shape")
        return Fp_val

    if is_function(X0):
        X0 = (X0,)

    if converged is None:
        def converged(it, F_old, F_new, X_new, G_new, S, Y):
            return False
    else:
        converged_arg = converged

        def converged(it, F_old, F_new, X_new, G_new, S, Y):
            return converged_arg(it, F_old, F_new,
                                 X_new[0] if len(X_new) == 1 else X_new,
                                 G_new[0] if len(G_new) == 1 else G_new,
                                 S[0] if len(S) == 1 else S,
                                 Y[0] if len(Y) == 1 else Y)

    if (H_0 is None and M_inv is None) and M is not None:
        raise OptimizationException("If M is supplied, then H_0 or M_inv must "
                                    "be supplied")
    if (H_0 is not None or M_inv is not None) and M is None:
        raise OptimizationException("If H_0 or M_inv are supplied, then M "
                                    "must be supplied")

    if H_0 is None:
        def H_0(*X):
            return X  # copy not required
    else:
        H_0 = wrapped_action(H_0)

    if M is None:
        def M(*X):
            return X  # copy not required
    else:
        M = wrapped_action(M)

    if M_inv is None:
        M_inv = H_0
    else:
        M_inv = wrapped_action(M_inv)

    if comm is None:
        comm = function_comm(X0[0])

    X = functions_copy(X0)
    del X0
    if old_F_val is None:
        old_F_val = F(*X)
    old_Fp_val = functions_copy(Fp(*X))
    old_Fp_norm_sq = abs(functions_inner(M_inv(*old_Fp_val), old_Fp_val))

    H_approx = H_approximation(m=m,
                               skip_atol=skip_atol, skip_rtol=skip_rtol,
                               M=M, M_inv=M_inv)
    if theta_scale and delta is not None:
        if block_theta_scale and len(old_Fp_val) > 1:
            old_M_inv_Fp = M_inv(*old_Fp_val)
            assert len(old_Fp_val) == len(old_M_inv_Fp)
            theta = [
                np.sqrt(abs(function_inner(old_M_inv_Fp[i], old_Fp_val[i])))
                / delta
                for i in range(len(old_Fp_val))]
            del old_M_inv_Fp
        else:
            theta = np.sqrt(old_Fp_norm_sq) / delta
    else:
        theta = 1.0

    it = 0
    conv = None
    reason = None
    logger.info(f"L-BFGS: Iteration {it:d}, "
                f"F calls {F_calls[0]:d}, "
                f"Fp calls {Fp_calls[0]:d}, "
                f"functional value {old_F_val:.6e}")
    while True:
        logger.debug(f"  Gradient norm = {np.sqrt(old_Fp_norm_sq):.6e}")
        if g_atol is not None and old_Fp_norm_sq <= g_atol * g_atol:
            conv = True
            reason = "g_atol reached"
            break

        minus_P = H_approx.action(old_Fp_val, H_0=H_0, theta=theta)
        if is_function(minus_P):
            minus_P = (minus_P,)
        alpha, old_Fp_val_rank0, new_F_val, new_Fp_val, new_Fp_val_rank0 = line_search(  # noqa: E501
            F, Fp, X, minus_P, c1=c1, c2=c2,
            old_F_val=old_F_val, old_Fp_val=old_Fp_val,
            line_search_rank0=line_search_rank0,
            line_search_rank0_kwargs=line_search_rank0_kwargs,
            comm=comm)
        if is_function(new_Fp_val):
            new_Fp_val = (new_Fp_val,)
        if alpha is None:
            if it == 0:
                raise OptimizationException("L-BFGS: Line search failure -- "
                                            "consider changing l-bfgs 'delta_lbfgs' value")
            logger.warning("L-BFGS: Line search failure -- resetting "
                           "Hessian inverse approximation")
            H_approx.reset()

            if theta_scale and delta is not None:
                if block_theta_scale and len(old_Fp_val) > 1:
                    old_M_inv_Fp = M_inv(*old_Fp_val)
                    assert len(old_Fp_val) == len(old_M_inv_Fp)
                    theta = [
                        np.sqrt(abs(function_inner(old_M_inv_Fp[i], old_Fp_val[i])))
                        / delta
                        for i in range(len(old_Fp_val))]
                    del old_M_inv_Fp
                else:
                    theta = np.sqrt(old_Fp_norm_sq) / delta
            else:
                theta = 1.0

            minus_P = H_approx.action(old_Fp_val, H_0=H_0, theta=theta)
            if is_function(minus_P):
                minus_P = (minus_P,)
            alpha, old_Fp_val_rank0, new_F_val, new_Fp_val, new_Fp_val_rank0 = line_search(  # noqa: E501
                F, Fp, X, minus_P, c1=c1, c2=c2,
                old_F_val=old_F_val, old_Fp_val=old_Fp_val,
                line_search_rank0=line_search_rank0,
                line_search_rank0_kwargs=line_search_rank0_kwargs,
                comm=comm)
            if is_function(new_Fp_val):
                new_Fp_val = (new_Fp_val,)
            if alpha is None:
                raise OptimizationException("L-BFGS: Line search failure")

        if new_F_val > old_F_val + c1 * alpha * old_Fp_val_rank0:
            raise OptimizationException("L-BFGS: Armijo condition not "
                                        "satisfied")
        if new_Fp_val_rank0 < c2 * old_Fp_val_rank0:
            raise OptimizationException("L-BFGS: Curvature condition not "
                                        "satisfied")
        if abs(new_Fp_val_rank0) > c2 * abs(old_Fp_val_rank0):
            logger.warning("L-BFGS: Strong curvature condition not satisfied")

        S = functions_new(minus_P)
        functions_axpy(S, -alpha, minus_P)
        functions_axpy(X, 1.0, S)

        Y = functions_copy(new_Fp_val)
        functions_axpy(Y, -1.0, old_Fp_val)

        S_inner_Y, S_Y_added, S_Y_removed = H_approx.append(S, Y, remove=True)
        if S_Y_added:
            if theta_scale:
                H_0_Y = H_0(*Y)
                if block_theta_scale and len(Y) > 1:
                    assert len(S) == len(Y)
                    assert len(S) == len(H_0_Y)
                    theta = [abs(function_inner(H_0_y, y) / function_inner(s, y))
                             for s, y, H_0_y in zip(S, Y, H_0_Y)]

                else:
                    theta = functions_inner(H_0_Y, Y) / S_inner_Y
                del H_0_Y

        else:
            logger.warning(f"L-BFGS: Iteration {it + 1:d}, small or negative "
                           f"inner product {S_inner_Y:.6e} -- update skipped")
        del S_Y_removed

        it += 1
        logger.info(f"L-BFGS: Iteration {it:d}, "
                    f"F calls {F_calls[0]:d}, "
                    f"Fp calls {Fp_calls[0]:d}, "
                    f"functional value {new_F_val:.6e}")
        if s_atol is not None:
            s_norm_sq = abs(functions_inner(S, M(*S)))
            logger.debug(f"  Change norm = {np.sqrt(s_norm_sq):.6e}")
            if s_norm_sq <= s_atol * s_atol:
                conv = True
                reason = "s_atol reached"
                break
        if converged(it, old_F_val, new_F_val, X, new_Fp_val, S, Y):
            conv = True
            reason = "converged"
            break

        if it >= max_its:
            conv = False
            reason = "max_its reached"
            break

        old_F_val = new_F_val
        old_Fp_val = new_Fp_val
        del new_F_val, new_Fp_val, new_Fp_val_rank0
        old_Fp_norm_sq = abs(functions_inner(M_inv(*old_Fp_val), old_Fp_val))

    assert conv is not None
    assert reason is not None
    return (X[0] if len(X) == 1 else X,
            it, conv, reason, F_calls[0], Fp_calls[0],
            H_approx)


def minimize_l_bfgs(forward, M0, m, s_atol, g_atol, J0=None, manager=None,
                    **kwargs):
    if not isinstance(M0, Sequence):
        (x,), optimization_data = minimize_l_bfgs(
            forward, (M0,), m, s_atol, g_atol, J0=J0, manager=manager,
            **kwargs)
        return x, optimization_data

    M0 = [m0 if is_function(m0) else m0.m() for m0 in M0]

    if manager is None:
        manager = _manager()

    M = [function_new(m0, static=function_is_static(m0),
                      cache=function_is_cached(m0),
                      checkpoint=function_is_checkpointed(m0))
         for m0 in M0]

    last_F = [None, None, None]
    if J0 is not None:
        last_F[0] = functions_copy(M0)
        last_F[1] = M0
        last_F[2] = J0

    @restore_manager
    def F(*X, force=False):
        if not force and last_F[0] is not None:
            change_norm = 0.0
            assert len(X) == len(last_F[0])
            for m, last_m in zip(X, last_F[0]):
                change = function_copy(m)
                function_axpy(change, -1.0, last_m)
                change_norm = max(change_norm, function_linf_norm(change))
            if change_norm == 0.0:
                return last_F[2].value()

        last_F[0] = functions_copy(X)
        functions_assign(M, X)
        clear_caches(*M)

        set_manager(manager)
        manager.reset()
        manager.stop()
        clear_caches()

        last_F[1] = M
        manager.start()
        last_F[2] = forward(last_F[1])
        manager.stop()

        return last_F[2].value()

    def Fp(*X):
        F(*X, force=last_F[1] is None)
        dJ = manager.compute_gradient(last_F[2], last_F[1])
        if manager._cp_method not in ["memory", "periodic_disk"]:
            last_F[1] = None
        return dJ

    X, its, conv, reason, F_calls, Fp_calls, H_approx = l_bfgs(
        F, Fp, M0, m, s_atol, g_atol, comm=manager.comm(), **kwargs)
    if is_function(X):
        X = (X,)

    return X, (its, conv, reason, F_calls, Fp_calls, H_approx)
