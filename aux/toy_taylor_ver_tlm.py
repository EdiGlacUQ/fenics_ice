import sys
from fenics import *
sys.path.insert(0,'/home/fenics/shared/dolfin_adjoint_custom/python')
from tlm_adjoint import *
from IPython import embed

start_annotating()

class toy_object:
    def __init__(self, mesh_in):
        self.mesh = mesh_in
        self.Q = FunctionSpace(mesh, 'CG', 1)
        self.test, self.trial = TestFunction(self.Q), TrialFunction(self.Q)

    def taylor_ver(self, alpha_in, annotate_flag=False):

        #self.alpha = project(alpha_in, self.Q, annotate=annotate_flag)

        self.alpha = Function(self.Q, name='alpha')
        eq = EquationSolver(inner(self.trial, self.test) * dx == inner(alpha_in, self.test) * dx, self.alpha)
        eq.solve()

        #self.J = inner(self.alpha,self.alpha)*dx Fails
        #self.Jf = Functional(self.J)

        self.J = inner(self.alpha,self.alpha)*dx
        self.Jf = Functional()
        self.Jf.assign(self.J)
        return self.Jf

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
dJ = compute_gradient(toy.Jf, cc)
ddJ= Hessian(toy.taylor_ver)
direction = interpolate(Constant(1.0),Q)
minconv = taylor_test(toy.taylor_ver, cc, assemble(toy.J), dJ, seed = 1e-3, size = 4)
