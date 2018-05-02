#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import petsc4py.PETSc as PETSc
import slepc4py.SLEPc as SLEPc

from fenics import *
import numpy as np
from scipy.linalg import  lu, qr, svd, eigh
from IPython import embed
import time

import unittest

# First written 2018-03-01
def slepsceig(N, A_action, hermitian = False, N_eigenvalues = None,
  which = SLEPc.EPS.Which.LARGEST_MAGNITUDE, tolerance = 1.0e-12, comm = None):
  """
  Matrix-free eigendecomposition using SLEPc via slepc4py, loosely following the
  slepc4py 3.6.0 demo demo/ex3.py.

  Arguments:

  N              Problem size.
  A_action       Function handle accepting a single array of shape (N,) and
                 returning an array of shape (N,).
  hermitian      (Optional) Whether the matrix is Hermitian.
  N_eigenvalues  (Optional) Number of eigenvalues to attempt to find. Defaults
                 to a full eigendecomposition.
  which          (Optional) Which eigenvalues to find.
  tolerance      (Optional) Tolerance, using SLEPc.EPS.Conv.EIG convergence
                 criterion.
  comm           (Optional) PETSc communicator.

  Returns:

  A tuple (lam, v), where lam is a vector of eigenvalues, and v is a matrix
  whose columns contain the corresponding eigenvectors.
  """

  class PythonMatrix(object):
    def __init__(self, action):
      self._action = action

    def mult(self, A, x, y):
      y[:] = self._action(x)

  A_matrix = PETSc.Mat().createPython((N, N), PythonMatrix(A_action), comm = comm)
  A_matrix.setUp()

  esolver = SLEPc.EPS().create()
  esolver.setProblemType(getattr(SLEPc.EPS.ProblemType, "HEP" if hermitian else "NHEP"))
  esolver.setOperators(A_matrix)
  esolver.setWhichEigenpairs(which)
  esolver.setDimensions(nev = N if N_eigenvalues is None else N_eigenvalues,
                        ncv = SLEPc.DECIDE, mpd = SLEPc.DECIDE)
  #esolver.setConvergenceTest(SLEPc.EPS.Conv.EIG)
  esolver.setTolerances(tol = tolerance, max_it = SLEPc.DECIDE)
  esolver.setUp()

  esolver.solve()

  lam = np.empty(esolver.getConverged(), dtype = np.float64 if hermitian else np.complex64)
  v = np.empty([N, lam.shape[0]], dtype = np.float64 if hermitian else np.complex64)
  v_r, v_i = A_matrix.getVecRight(), A_matrix.getVecRight()
  for i in xrange(v.shape[1]):
    lam_i = esolver.getEigenpair(i, v_r, v_i)
    if hermitian:
      lam[i] = lam_i.real
      v[:, i] = v_r.getArray()
    else:
      lam[i] = lam_i
      v[:, i] = v_r.getArray() + 1.0j * v_i.getArray()

  return lam, v


def randeig(A, k=6, n_iter=4, l=None):

    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert m == n
    assert k > 0
    assert k <= n
    assert n_iter >= 0
    assert l >= k

    # Apply A to a random matrix, obtaining Q.
    print('Forming Q')
    t0 = time.time()
    Q = matmult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
    t1 = time.time()
    print('Completed in: {0}'.format(t1-t0))

    # Form a matrix Q whose columns constitute a well-conditioned basis
    # for the columns of the earlier Q.
    if n_iter == 0:
        (Q, _) = qr(Q, mode='economic')
    if n_iter > 0:
        (Q, _) = lu(Q, permute_l=True)

    # Conduct normalized power iterations.
    for it in range(n_iter):
        print('Normalized Power Iteration: {0}'.format(it))
        t0 = time.time()
        Q = matmult(A, Q)

        if it + 1 < n_iter:
            (Q, _) = lu(Q, permute_l=True)
        else:
            (Q, _) = qr(Q, mode='economic')
        t1 = time.time()
        print('Completed in: {0}'.format(t1-t0))

    # Eigendecompose Q'*A*Q to obtain approximations to the eigenvalues
    # and eigenvectors of A.
    print('Eigendecomposition of Q\'*A*Q')
    t0 = time.time()
    R = Q.conj().T.dot(matmult(A, Q))
    R = (R + R.conj().T) / 2
    (d, V) = eigh(R)
    V = Q.dot(V)
    t1 = time.time()
    print('Completed in: {0}'.format(t1-t0))

    # Retain only the entries of d with the k greatest absolute values
    # and the corresponding columns of V.
    idx = abs(d).argsort()[-k:][::-1]
    return d[idx], V[:, idx]


def matmult(A,B):
    return np.array(map(A.apply,B.T)).T

class HessWrapper(object):

    def __init__(self,action, cntrl):
        self._action = action
        self.xfn = Function(cntrl.function_space())
        n = cntrl.vector().size()
        self.shape = (n,n)

    def apply(self, x):
        if type(x) is not np.ndarray:
            x = x.getArray()
        self.xfn.vector().set_local(x)
        self.xfn.vector().apply(str('insert'))
        return self._action(self.xfn).vector().get_local()


class eigendecomposition_unittests(unittest.TestCase):
  def test_HEP(self):
    A = np.array([[3.0, 0.0], [0.0, -10.0]], dtype = np.float64)
    lam, v = eig(A.shape[0], lambda x : np.dot(A, x), hermitian = True)
    self.assertAlmostEqual(min(lam), -10.0)
    self.assertAlmostEqual(max(lam), 3.0)
    v_0, v_1 = v[:, 0], v[:, 1]
    self.assertAlmostEqual(np.dot(v_0, v_0), 1.0)
    self.assertAlmostEqual(np.dot(v_0, v_1), 0.0)
    self.assertAlmostEqual(np.dot(v_1, v_1), 1.0)

    for i in range(100):
      A = np.random.random((10, 10))
      A = 0.5 * (A + A.T);
      lam, v = eig(A.shape[0], lambda x : np.dot(A, x), hermitian = True)
      self.assertAlmostEqual(max(abs(np.array(sorted(np.linalg.eig(A)[0]), dtype = np.float64) -
                                     np.array(sorted(lam)))), 0.0)

  def test_NHEP(self):
    for i in range(100):
      A = np.random.random((10, 10))
      lam, v = eig(A.shape[0], lambda x : np.dot(A, x))
      self.assertAlmostEqual(max(abs(np.array(sorted(np.linalg.eig(A)[0]), dtype = np.complex64) -
                                     np.array(sorted(lam)))), 0.0)

if __name__ == "__main__":
  np.random.seed(1626728)
  unittest.main()
