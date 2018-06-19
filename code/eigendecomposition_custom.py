#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../../dolfin_adjoint_custom/python/')
from tlm_adjoint import *

import numpy as np
from scipy.linalg import  lu, qr, eigh
from IPython import embed
import time
import petsc4py


def rand_hep(space, A_action, k=6, n_iter=4, l=None):
  A = action_wrapper(A_action,space)

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

  # Retain K greatest eigenpairs
  idx = abs(d).argsort()[-k:][::-1]
  return d[idx], V[:, idx]


def matmult(A,B):
  return np.array(list(map(A.apply,B.T))).T

class action_wrapper(object):

  def __init__(self,action, space):
      self._action = action
      self.xfn = Function(space)
      n = self.xfn.vector().size()
      self.shape = (n,n)

  def apply(self, x):
      if type(x) is not np.ndarray:
          x = x.getArray()
      self.xfn.vector().set_local(x)
      self.xfn.vector().apply(str('insert'))
      return self._action(self.xfn)
