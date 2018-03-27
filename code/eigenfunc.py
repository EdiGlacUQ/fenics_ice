from fenics import *
import numpy as np
from scipy.linalg import  lu, qr, svd, eigh
from IPython import embed
import time



def eigens(A, k=6, n_iter=4, l=None):
    """
    Eigendecomposition of a SELF-ADJOINT matrix.
    Constructs a nearly optimal rank-k approximation V diag(d) V' to a
    SELF-ADJOINT matrix A, using n_iter normalized power iterations,
    with block size l, started with an n x l random matrix, when A is
    n x n; the reference EGS_ below explains "nearly optimal." k must
    be a positive integer <= the dimension n of A, n_iter must be a
    nonnegative integer, and l must be a positive integer >= k.
    The rank-k approximation V diag(d) V' comes in the form of an
    eigendecomposition -- the columns of V are orthonormal and d is a
    vector whose entries are real-valued and their absolute values are
    nonincreasing. V is n x k and len(d) = k, when A is n x n.
    Increasing n_iter or l improves the accuracy of the approximation
    V diag(d) V'; the reference EGS_ below describes how the accuracy
    depends on n_iter and l. Please note that even n_iter=1 guarantees
    superb accuracy, whether or not there is any gap in the singular
    values of the matrix A being approximated, at least when measuring
    accuracy as the spectral norm || A - V diag(d) V' || of the matrix
    A - V diag(d) V' (relative to the spectral norm ||A|| of A).
    Notes
    -----
    THE MATRIX A MUST BE SELF-ADJOINT.
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.
    The user may ascertain the accuracy of the approximation
    V diag(d) V' to A by invoking diffsnorms(A, numpy.diag(d), V).
    Parameters
    ----------
    A : array_like, shape (n, n)
        matrix being approximated
    k : int, optional
        rank of the approximation being constructed;
        k must be a positive integer <= the dimension of A, and
        defaults to 6
    n_iter : int, optional
        number of normalized power iterations to conduct;
        n_iter must be a nonnegative integer, and defaults to 4
    l : int, optional
        block size of the normalized power iterations;
        l must be a positive integer >= k, and defaults to k+2
    Returns
    -------
    d : ndarray, shape (k,)
        vector of length k in the rank-k approximation V diag(d) V'
        to A, such that its entries are real-valued and their absolute
        values are nonincreasing
    V : ndarray, shape (n, k)
        n x k matrix in the rank-k approximation V diag(d) V' to A,
        where A is n x n
    Examples
    --------
    >>> from fbpca import diffsnorms, eigens
    >>> from numpy import diag
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(2, 100))
    >>> A = A.T.dot(A)
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (d, V) = eigens(A, 2)
    >>> err = diffsnorms(A, diag(d), V)
    >>> print(err)
    This example produces a rank-2 approximation V diag(d) V' to A
    such that the columns of V are orthonormal, and the entries of d
    are real-valued and their absolute values are nonincreasing.
    diffsnorms(A, diag(d), V) outputs an estimate of the spectral norm
    of A - V diag(d) V', which should be close to the machine
    precision.
    References
    ----------
    .. [EGS] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic
             algorithms for constructing approximate matrix
             decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009
             (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    See also
    --------
    diffsnorms, eigenn, pca
    """

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
