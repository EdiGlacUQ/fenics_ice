#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import petsc4py.PETSc as PETSc
import slepc4py.SLEPc as SLEPc

import numpy
import unittest

# First written 2018-03-01
def eig(N, A_action, hermitian = False, N_eigenvalues = None,
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
 # esolver.setConvergenceTest(SLEPc.EPS.Conv.EIG)
  esolver.setTolerances(tol = tolerance, max_it = SLEPc.DECIDE)
  esolver.setUp()

  esolver.solve()

  lam = numpy.empty(esolver.getConverged(), dtype = numpy.float64 if hermitian else numpy.complex64)
  v = numpy.empty([N, lam.shape[0]], dtype = numpy.float64 if hermitian else numpy.complex64)
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

        
class eigendecomposition_unittests(unittest.TestCase):
  def test_HEP(self):
    A = numpy.array([[3.0, 0.0], [0.0, -10.0]], dtype = numpy.float64)
    lam, v = eig(A.shape[0], lambda x : numpy.dot(A, x), hermitian = True)
    self.assertAlmostEqual(min(lam), -10.0)
    self.assertAlmostEqual(max(lam), 3.0)
    v_0, v_1 = v[:, 0], v[:, 1]
    self.assertAlmostEqual(numpy.dot(v_0, v_0), 1.0)
    self.assertAlmostEqual(numpy.dot(v_0, v_1), 0.0)
    self.assertAlmostEqual(numpy.dot(v_1, v_1), 1.0)

    for i in range(100):
      A = numpy.random.random((10, 10))
      A = 0.5 * (A + A.T);
      lam, v = eig(A.shape[0], lambda x : numpy.dot(A, x), hermitian = True)
      self.assertAlmostEqual(max(abs(numpy.array(sorted(numpy.linalg.eig(A)[0]), dtype = numpy.float64) -
                                     numpy.array(sorted(lam)))), 0.0)

  def test_NHEP(self):
    for i in range(100):
      A = numpy.random.random((10, 10))
      lam, v = eig(A.shape[0], lambda x : numpy.dot(A, x))
      self.assertAlmostEqual(max(abs(numpy.array(sorted(numpy.linalg.eig(A)[0]), dtype = numpy.complex64) -
                                     numpy.array(sorted(lam)))), 0.0)

if __name__ == "__main__":
  numpy.random.seed(1626728)
  unittest.main()
