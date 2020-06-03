import sys
# sys.path.insert(0,'../../tlm_adjoint/python/')

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
        self.A.init_vector(self.tmp1, 0)
        self.A.init_vector(self.tmp2, 1)

        # Inverse Root Lumped mass matrix:
        # NB lots of misnomers here - easier than constructing fresh objects
        mass_action_form = action(var_m, Constant(1))
        lump_diag = assemble(mass_action_form)
        root_lump_diag = (lump_diag.get_local()) ** 0.5
        inv_root_lump_diag = 1.0 / root_lump_diag
        inv_lump_diag = 1.0 / lump_diag.get_local()

        self.M_irl = assemble(var_m)  # dummy, zeroed
        self.M_rl = assemble(var_m)  # dummy, zeroed
        self.M_l = assemble(var_m)  # dummy, zeroed
        self.M_il = assemble(var_m)  # dummy, zeroed

        self.M_irl.zero()
        self.M_rl.zero()
        self.M_l.zero()
        self.M_il.zero()

        # TODO - don't need all these (M_rl specifically?)
        self.M_l.set_diagonal(lump_diag)
        lump_diag.set_local(inv_root_lump_diag)
        self.M_irl.set_diagonal(lump_diag)
        lump_diag.set_local(root_lump_diag)
        self.M_rl.set_diagonal(lump_diag)
        lump_diag.set_local(inv_lump_diag)
        self.M_il.set_diagonal(lump_diag)

    def action(self, x, y):
        """
        LM^-1L
        """
        self.A.mult(x, self.tmp1) #tmp1 = Ax
        self.M_solver.solve(self.tmp2, self.tmp1) #Atmp2 = tmp1
        self.A.mult(self.tmp2,self.tmp1)
        y.set_local(self.tmp1.get_local())
        y.apply("insert")

    def inv_action(self, x, y):
        """
        L^-1 M L^-1
        """
        self.A_solver.solve(self.tmp1, x)
        self.M.mult(self.tmp1, self.tmp2)
        self.A_solver.solve(self.tmp1, self.tmp2)
        y.set_local(self.tmp1.get_local())
        y.apply("insert")

    def approx_action(self, x, y):
        """
        L M_lump^-1L
        """
        self.A.mult(x, self.tmp1) #tmp1 = Ax
        self.M_il.mult(self.tmp1, self.tmp2)
        self.A.mult(self.tmp2,self.tmp1)
        y.set_local(self.tmp1.get_local())
        y.apply("insert")

    def approx_root_inv_action(self, x, y):
        """
        L^-1 M_lump^-1/2
        Used as a GHEP preconditioner
        eigendecomposition_transformation.tex
        """
        # embed()

        if not isinstance(x, PETScVector):
            x_tmp = PETScVector(x)
        else:
            x_tmp = x

        if not isinstance(y, PETScVector):
            y_tmp = PETScVector(y)
        else:
            y_tmp = y

        self.M_irl.mult(x_tmp, self.tmp1)
        self.A_solver.solve(y_tmp, self.tmp1)
