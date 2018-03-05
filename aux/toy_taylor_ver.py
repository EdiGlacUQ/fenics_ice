from fenics import *
from dolfin_adjoint import *


class toy_object:
    def __init__(self, mesh_in):
        self.mesh = mesh_in
        self.Q = FunctionSpace(mesh, 'CG', 1)

    def taylor_ver(self, alpha_in, annotate_flag=False):

        self.alpha = project(alpha_in, self.Q, annotate=annotate_flag)
        self.J =  inner(self.alpha,self.alpha)*dx
        return assemble(self.J)

#Basic mesh and elements
n = 100
mesh = UnitSquareMesh(n, n)
Q = FunctionSpace(mesh, 'CG', 1)

#Initialize the control
alpha0 = project(Constant(1.0),  Q, annotate=False)
cc = Control(alpha0)

#Initialize the toy object
toy = toy_object(mesh)

#Taylor verification
toy.taylor_ver(alpha0,annotate_flag=True)
dJ = compute_gradient(Functional(toy.J), cc, forget = False)
ddJ = hessian(Functional(toy.J), cc)

#minconv begins to fail at 1e-5
minconv = taylor_test(toy.taylor_ver, cc, assemble(toy.J), dJ, HJm = ddJ, seed = 1e-1, size = 4)
