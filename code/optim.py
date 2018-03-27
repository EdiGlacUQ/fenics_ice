from fenics import *
import numpy as np
from numpy import random
from scipy.sparse.linalg import cg
import unittest
from IPython import embed

def conjgrad(A, b, x0 = None, max_iter = np.Inf, tol=1e-10, verbose = True):
    fs = b.function_space()
    x, p, r = Function(fs), Function(fs), Function(fs)

    x.assign(Constant(0.0) if x0 is None else x0)
    p.assign(Constant(0.0))
    r.assign(Constant(0.0))

    max_iter = min(100*b.vector().size(), max_iter)

    r.assign(b-A(x))
    p.assign(r)
    rs = innerprod(r, r)

    if np.sqrt(rs) < tol:
        print '''Norm of residual is less than tolerance (%1.2e) \n
                Converged in %i iterations ''' % (tol, 0)
        return x


    for i in xrange(max_iter):

        print '\nIteration: %i' % i

        Ap = A(p)
        alpha = rs / innerprod(p, Ap)
        axpy(x, alpha, p)
        axpy(r, -alpha, Ap)
        rs_new = innerprod(r, r)

        #Check stopping criterion
        if np.sqrt(rs_new) < tol:
            print '''Norm of residual is less than tolerance (%1.2e)
            Converged in %i iterations ''' % (tol, i)
            return x

        elif i+1==max_iter:
            print '''Maximum number of iterations reached (%i)
            Norm of residual is (%1.2e)''' % (max_iter, np.sqrt(rs_new))
            return x

        p.assign( (rs_new / rs) * p + r)
        rs = rs_new

        if verbose:
            print 'Iteration: %i, Norm of residual: %1.2e' % (i,np.sqrt(rs_new))



def conjgrad_nparray(A, b, x0 = None, max_iter = np.Inf, tol=1e-10, verbose = True):

    x = np.zeros(b.size)
    p = np.zeros(b.size)
    r = np.zeros(b.size)

    max_iter = min(100*b.size, max_iter)


    r = b - A.dot(x)
    p = r
    rs = np.inner(r,r)



    if np.sqrt(rs) < tol:
        print '''Norm of residual is less than tolerance (%1.2e) \n
                Converged in %i iterations ''' % (tol, 0)
        return x


    for i in range(max_iter):

        Ap = A.dot(p)
        alpha = rs / np.inner(p, Ap)
        x = x + alpha*p
        r = r - alpha*Ap
        rs_new = np.inner(r,r)

        #Check stopping criterion
        if np.sqrt(rs_new) < tol:
            print '''Norm of residual is less than tolerance (%1.2e)
            Converged in %i iterations ''' % (tol, i)
            return x

        elif i+1==max_iter:
            print '''Maximum number of iterations reached (%i)
            Norm of residual is (%1.2e)''' % (max_iter, np.sqrt(rs_new))
            return x

        p = r + (rs_new / rs) * p
        rs = rs_new

        if verbose:
            print 'Iteration: %i, Norm of residual: %1.2e' % (i,np.sqrt(rs_new))



def innerprod(x, y):
    inner = 0.0
    x = x.vector()
    y = y.vector()
    inner += x.inner(y)
    return inner

def axpy(x, alpha, y):
    x = x.vector()
    y = y.vector()
    x.axpy(alpha, y)


class cg_unittests(unittest.TestCase):
    def test_solution(self):
        random.seed(1626728)
        N = 2
        mesh = UnitSquareMesh(N, N)
        V = FunctionSpace(mesh,'P',1)
        b = Function(V)

        A = random.rand(b.vector().size(),b.vector().size())
        pdmatrix = np.dot(A,A.transpose())
        pdmatrix_obj = PythonHess(pdmatrix)

        b_array = random.rand(b.vector().size())
        b.vector().set_local(b_array)

        x0 = cg(pdmatrix,b_array)[0]
        x1 = conjgrad_nparray(pdmatrix,b_array)
        x2 = conjgrad(pdmatrix_obj.action, b)

        self.assertAlmostEqual(0.0, np.linalg.norm(x0 - x2.vector().array()))
        self.assertAlmostEqual(0.0, np.linalg.norm(x0 - x1))
        self.assertAlmostEqual(0.0, np.linalg.norm(x2.vector().array() - x1, np.inf))

class PythonHess(object):
    def __init__(self, np_matrix):
        self.m = np_matrix

    def action(self,x):
        y = x.copy(deepcopy=True)
        y.vector().set_local(self.m.dot(x.vector().array()))
        return y


if __name__ == "__main__":
    random.seed(1626728)
    unittest.main()
