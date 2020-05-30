import sys
# sys.path.insert(0,'../../tlm_adjoint/python/')
import numpy as np
from numpy import random
from fenics_ice.sqrt_mass_matrix_action import A_root_action
from dolfin import *
from tlm_adjoint import *

from IPython import embed

class error_propogator(object):

    def __init__():
      pass

class laplacian(object):

    def __init__(self, delta, gamma, space):

        test, trial = TestFunction(space), TrialFunction(space)

        var_m = inner(test,trial)*dx
        var_n = inner(grad(test), grad(trial))*dx


        self.M = assemble(var_m)
        self.M_solver = KrylovSolver("cg", "sor")
        self.M_solver.parameters.update({"absolute_tolerance":1.0e-32,
                                   "relative_tolerance":1.0e-14})
        self.M_solver.set_operator(self.M)


        self.A = assemble(delta*var_m + gamma*var_n)
        self.A_solver = KrylovSolver("cg", "sor")
        self.A_solver.set_operator(self.A)

        self.tmp1, self.tmp2 = Function(space), Function(space)

        self.tmp1, self.tmp2 = Vector(), Vector()
        self.tmp3 = Vector()
        self.A.init_vector(self.tmp1, 0)
        self.A.init_vector(self.tmp2, 1)


    def action(self, x, y):      # A M-1 A
        self.A.mult(x, self.tmp1) #tmp1 = Ax
        self.M_solver.solve(self.tmp2, self.tmp1) #Atmp2 = tmp1
        self.A.mult(self.tmp2,self.tmp1)
        y.set_local(self.tmp1.get_local())
        y.apply("insert")

    def inv_action(self, x, y):
        self.A_solver.solve(self.tmp1, x)
        self.M.mult(self.tmp1, self.tmp2)
        self.A_solver.solve(self.tmp1, self.tmp2)
        y.set_local(self.tmp1.get_local())
        y.apply("insert")
        
    def sqrt_action(self, x, y):     # sqrt of inv cov: Gamma -1 Gamma 1/2
                                     #                  A M-1 A A-1 M1/2
                                     #                  A M-1 M1/2
        M_norm = self.M.norm("linf")
        self.tmp1, terms = A_root_action(self.M, x, tol=1.0e-16, beta=M_norm)
        self.M_solver.solve(self.tmp2, self.tmp1) 
        self.A.mult(self.tmp2,self.tmp3)
        y.set_local(self.tmp3.get_local())
        y.apply("insert")
        
    def sqrt_inv_action(self, x, y): # sqrt of cov: Gamma 1/2
                                     #                  A-1 M1/2
        M_norm = self.M.norm("linf")        
        self.tmp1, terms = A_root_action(self.M, x, tol=1.0e-16, beta=M_norm)
        self.A_solver.solve(self.tmp2, self.tmp1)
        y.set_local(self.tmp2.get_local())
        y.apply("insert")
        
    def sample_prior(self,x):        # sqrt cov: A-1 M^1/2
        shp = np.shape(x.vec().array)
        np.random.seed()
        x.vec().array = random.normal(np.zeros(shp),
                     np.ones(shp),shp)
        M_norm = self.M.norm("linf")
        self.tmp1, terms = A_root_action(self.M, x, tol=1.0e-16, beta=M_norm)
        self.A_solver.solve(self.tmp2, self.tmp1)
        x.set_local(self.tmp2.get_local())
        x.apply("insert")
        
        
    def sample_inv_prior(self,x):        # sqrt inv cov: A M-1 M1/2
        shp = np.shape(x.vec().array)
        np.random.seed()
        x.vec().array = random.normal(np.zeros(shp),
                     np.ones(shp),shp)
        M_norm = self.M.norm("linf")
        self.tmp1, terms = A_root_action(self.M, x, tol=1.0e-16, beta=M_norm)
        self.M_solver.solve(self.tmp2, selpf.tmp1) 
        self.A.mult(self.tmp2,self.tmp3)
        x.set_local(self.tmp3.get_local())
        x.apply("insert")