#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import TestFunction, TrialFunction, assemble, dx, inner, split
import ufl
import numpy as np

__all__ = \
    [
        "A_root_action",
        "LumpedPCSqrtMassAction",
        "RootActionException"
    ]


class RootActionException(Exception):
    pass


def A_root_action(A, x, tol, beta=1.0, max_terms=1000):
    """
    Compute the action of the principal square root of the symmetric positive
    definite matrix A, as in
      N. J. Higham, "Functions of Matrices: Theory and Computation", SIAM,
      2008, (6.38)
    Arguments:
      A          A Matrix, or a callable of the form
                     def A(x):
                 where x is a Vector, returning A x as a Vector.
      x          Matrix action direction.
      tol        Absolute tolerance. Positive float.
      beta       Compute sqrt{beta} (A / beta)^{1/2} (s in Higham 2008
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
    alpha = [1, 2]

    j = 1
    while True:
        z.axpy(-1.0 / beta, A_action(z))
        change = y.copy()
        y.axpy(-alpha[0] / alpha[1], z)
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
        alpha[0] *= 2 * j - 3
        alpha[1] *= 2 * j

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
        M_L = assemble(sum(split(test), ufl.zero()) * dx) 

        if M is None:
            trial = TrialFunction(space)
            M = assemble(inner(trial, test) * dx)

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
