from fenics import *
import numpy as np
from scipy.linalg import  lu, qr, svd, eigh
from IPython import embed
import time



def eigens(A, k=6, n_iter=4, l=None):

    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert m == n
    assert k > 0
    assert k <= n
    assert n_iter >= 0
    assert l >= k


    #
    # Apply A to a random matrix, obtaining Q.
    #
    print 'Forming Q'
    t0 = time.time()
    Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
    t1 = time.time()
    print "Completed in: ", t1-t0



    #
    # Form a matrix Q whose columns constitute a well-conditioned basis
    # for the columns of the earlier Q.
    #
    if n_iter == 0:
        (Q, _) = qr(Q, mode='economic')
    if n_iter > 0:
        (Q, _) = lu(Q, permute_l=True)

    #
    # Conduct normalized power iterations.
    #
    for it in range(n_iter):
        print 'Normalized Power Iteration: ', it
        t0 = time.time()
        Q = mult(A, Q)

        if it + 1 < n_iter:
            (Q, _) = lu(Q, permute_l=True)
        else:
            (Q, _) = qr(Q, mode='economic')
        t1 = time.time()
        print "Completed in: ", t1-t0

    #
    # Eigendecompose Q'*A*Q to obtain approximations to the eigenvalues
    # and eigenvectors of A.
    #
    print 'Eigendecomposition of Q\'*A*Q'
    t0 = time.time()
    R = Q.conj().T.dot(mult(A, Q))
    R = (R + R.conj().T) / 2
    (d, V) = eigh(R)
    V = Q.dot(V)
    t1 = time.time()
    print "Completed in: ", t1-t0

    #
    # Retain only the entries of d with the k greatest absolute values
    # and the corresponding columns of V.
    #
    idx = abs(d).argsort()[-k:][::-1]
    return d[idx], V[:, idx]


def mult(A,B):
    return np.array(map(A.apply,B.T)).T

class HessWrapper(object):

    def __init__(self,action, cntrl):
        self._action = action
        self.xfn = Function(cntrl.function_space())
        n = cntrl.vector().size()
        self.shape = (n,n)

    def apply(self, x):
        self.xfn.vector().set_local(x)
        self.xfn.vector().apply('insert')
        return self._action(self.xfn).vector().get_local()
