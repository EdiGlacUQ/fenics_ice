#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import *

import numpy as np
import scipy.special as special
from IPython import embed

__all__ = \
    [
        "A_root_action",
        "LumpedPCSqrtMassAction",
        "RootActionException"
    ]


class RootActionException(Exception):
    pass


def coeff(k):
    # Binomial coefficient for sqrt(1 - x), similar to
    #   https://en.wikipedia.org/wiki/Binomial_coefficient#Binomial_coefficient_with_n_=_1/2
    #   [accessed 2020-04-22]
    # and derivable from the generating function for the central binomial
    # coefficients
    c = (special.comb(2 * k, k, exact=True), 0.0)
    for i in range(2 * k):
        if c[0] % 2 == 0:
            c = (c[0] // 2, c[1] / 2)
        else:
            c = (c[0] // 2, 0.5 + (c[1] / 2))
    return (c[0] / (2 * k - 1)) + (c[1] / (2 * k - 1))


def A_root_action(A, x, tol, beta=1.0, max_terms=1000):
    """
    Compute the action of the square root of the symmetric positive definite
    matrix A using binomial iteration, as in
      N. J. Higham, "Functions of Matrices: Theory and Computation", SIAM,
      2008, equation (6.38)
    Arguments:
      A          A Matrix, or a callable of the form
                     def A(x):
                 where x is a Vector, returning A x as a Vector.
      x          Matrix action direction.
      tol        Absolute tolerance. Positive float.
      beta       Compute \sqrt{beta} (A / beta)^{1/2} (s in Higham 2008
                 equation (6.39)). Positive float.
      max_its    Maximum number of terms in the binomial expansion. Integer,
                 >= 2.
    Returns:
      y, terms
    where y is a Vector containing the action, and terms is the number of
    terms added.
    """

    if callable(A):
        A_action = A
    else:
        def A_action(x):
            return A * x

    y = x.copy()
    z = x.copy()


    j = 1
    while True:
        z.axpy(-1.0 / beta, A_action(z))
        change = y.copy()
        y.axpy(-coeff(j), z)
        if np.isnan(y.sum()):
            raise RootActionException("NaN encountered")
        change.axpy(-1.0, y)
        change_norm = change.norm("linf") * np.sqrt(beta)
        # print(f"terms {j + 1:d}, norm = {change_norm:.6e}")
        if change_norm < tol:
            break
        j += 1
        if j >= max_terms:
            raise RootActionException("Maximum terms exceeded")

    return y * np.sqrt(beta), j + 1


class LumpedPCSqrtMassAction:
    def __init__(self, space, tol, beta=1.0, M=None):
        """
        Class for the calculation of matrix actions
            A x,
        where
            A = M_L^{1/2} (M_L^{-1/2} M M_L^{-1/2})^{1/2},
        so that
            A A^T = M,
        where M is the mass matrix and M_L the row-summed lumped mass matrix.
        The square root of M_L^{-1/2} M M_L^{-1/2} is computed using
        A_root_action.
        Arguments:
            space      The function space.
            tol, beta  As for A_root_action.
            M          (Optional) Mass matrix.
        """

        test = TestFunction(space)
        M_L = assemble(test * dx)

        if M is None:
            trial = TrialFunction(space)
            M = assemble(inner(test, trial) * dx)

        sqrt_M_L = M_L.copy()
        sqrt_M_L.set_local(np.sqrt(M_L.get_local()))
        sqrt_M_L.apply("insert")

        sqrt_M_L_inv = M_L.copy()
        sqrt_M_L_inv.set_local(1.0 / np.sqrt(M_L.get_local()))
        sqrt_M_L_inv.apply("insert")

        self._tol = tol
        self._beta = beta
        self._M = M
        self._sqrt_M_L = sqrt_M_L
        self._sqrt_M_L_inv = sqrt_M_L_inv

    def action(self, x):
        """
        Compute.
            A x,
        where
            A = M_L^{1/2} (M_L^{-1/2} M M_L^{-1/2})^{1/2},
        so that
            A A^T = M.
        Arguments:
            x  The matrix action direction.
        Returns:
          y, terms
        where y is a Vector containing the action, and terms is the number of
        terms added in A_root_action.
        """

        def transformed_M_action(x):
            return self._sqrt_M_L_inv * (self._M * (self._sqrt_M_L_inv * x))

        y, terms = A_root_action(transformed_M_action, x, tol=self._tol,
                                 beta=self._beta)
        y = self._sqrt_M_L * y
        return y, terms

    def inverse_action(self, x, solver):
        """
        Compute.
            B x,
        where
            B = M^{-1} A,
            A = M_L^{1/2} (M_L^{-1/2} M M_L^{-1/2})^{1/2},
        so that
            B B^T = M^{-1}.
        Arguments:
            x       Vector. The matrix action direction.
            solver  Linear solver, used to solve for y in
                        M y = A x.
        Returns:
          y, terms
        where y is a Vector containing the action, and terms is the number of
        terms added in A_root_action.
        """

        y_, terms = self.action(x)
        y = y_.copy()
        solver.solve(y, y_)
        return y, terms


def test_sqrt_mass():
    mesh = UnitSquareMesh(500, 500, "crossed")
    space = FunctionSpace(mesh, "Lagrange", 2)
    print(f"{space.dim():d} dofs")
    test, trial = TestFunction(space), TrialFunction(space)
    M = assemble(inner(test, trial) * dx)

    x = Function(space, name="x")
    x.interpolate(Expression("exp(x[0]) * exp(x[1])",
                             element=space.ufl_element()))
    x = x.vector()

    M_norm = M.norm("linf")
    y, terms = A_root_action(M, x, tol=1.0e-16, beta=M_norm)
    print(f"{terms:d} terms")
    z, terms = A_root_action(M, y, tol=1.0e-16, beta=M_norm)
    print(f"{terms:d} terms")

    z_ref = M * x
    z_error_norm = (z - z_ref).norm("linf")

    print(f'mass action inf norm = {z_ref.norm("linf"):.6e}')
    print(f'mass action inf norm error = {z_error_norm:.6e}')
    assert z_error_norm < 1.0e-14


def test_sample_p1_mass():
    # For a reproducible test
    np.random.seed(16292392)

    mesh = UnitIntervalMesh(2)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    M = assemble(inner(test, trial) * dx)
    M_solver = LUSolver(M)

    sqrt_M_action = LumpedPCSqrtMassAction(space, tol=1.0e-10, beta=2.0 / 3.0,
                                           M=M)

    M_inv = np.linalg.inv(M.array())

    y = Function(space, name="y").vector()
    assert space.dim() == y.local_size()  # Serial only test
    x_outer_sum = [np.zeros((space.dim(), space.dim()), dtype=np.float64), 0]
    for i in range(100000):
        y.set_local(np.random.randn(y.local_size()))
        y.apply("insert")
        x, terms = sqrt_M_action.inverse_action(y, M_solver)

        x_outer_sum[0] += np.outer(x.get_local(), x.get_local())
        x_outer_sum[1] += 1

        if (i + 1) % 250 == 0:
            print(f"sample {i + 1:d}")
            x_outer_mean = x_outer_sum[0] / x_outer_sum[1]
            print("M_inv =")
            print(M_inv)
            print("covariance =")
            print(x_outer_mean)
            error_norm = abs(M_inv - x_outer_mean).max()
            print(f"error norm = {error_norm:.6e}")

    x_outer_mean = x_outer_sum[0] / x_outer_sum[1]
    print("M_inv =")
    print(M_inv)
    print("covariance =")
    print(x_outer_mean)
    error_norm = abs(M_inv - x_outer_mean).max()
    print(f"error norm = {error_norm:.6e}")
    assert error_norm < 0.09
