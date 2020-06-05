from dolfin import *
from tlm_adjoint import *
from .decorators import count_calls, timer

class laplacian(object):

    def __init__(self, delta, gamma, space):

        self.space = space

        test, trial = TestFunction(space), TrialFunction(space)

        var_m = inner(test, trial) * dx
        var_n = inner(grad(test), grad(trial)) * dx

        self.M = assemble(var_m)
        self.M_solver = KrylovSolver("cg", "sor")
        self.M_solver.parameters.update({"absolute_tolerance": 1.0e-32,
                                         "relative_tolerance": 1.0e-14})
        self.M_solver.set_operator(self.M)

        self.A = assemble(delta * var_m + gamma * var_n)
        self.A_solver = KrylovSolver("cg", "sor")
        self.A_solver.parameters.update({"absolute_tolerance": 1.0e-32,
                                         "relative_tolerance": 1.0e-14})
        self.A_solver.set_operator(self.A)

        self.tmp1, self.tmp2 = Function(space), Function(space)

        self.tmp1, self.tmp2 = Vector(), Vector()
        self.A.init_vector(self.tmp1, 0)
        self.A.init_vector(self.tmp2, 1)

        # Inverse Root Lumped mass matrix (etc):
        # All stored as vectors for efficiency
        # NB: mass matrix here assumes no Dirichlet BC (OK as precond)
        mass_action_form = action(var_m, Constant(1))
        lump_diag = assemble(mass_action_form).get_local()
        root_lump_diag = (lump_diag) ** 0.5
        inv_root_lump_diag = 1.0 / root_lump_diag
        inv_lump_diag = 1.0 / lump_diag

        comm = self.A.mpi_comm()  # TODO - is this correct?
        dim = space.dim()

        self.M_l = Vector(comm, dim)
        self.M_l.set_local(lump_diag)
        self.M_l.apply("insert")

        self.M_il = Vector(comm, dim)
        self.M_il.set_local(inv_lump_diag)
        self.M_il.apply("insert")

        self.M_rl = Vector(comm, dim)
        self.M_rl.set_local(root_lump_diag)
        self.M_rl.apply("insert")

        self.M_irl = Vector(comm, dim)
        self.M_irl.set_local(inv_root_lump_diag)
        self.M_irl.apply("insert")

    def action(self, x, y):
        """
        LM^-1L
        """
        self.A.mult(x, self.tmp1)  # tmp1 = Ax
        self.M_solver.solve(self.tmp2, self.tmp1)  # Atmp2 = tmp1
        self.A.mult(self.tmp2, self.tmp1)
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
        self.A.mult(x, self.tmp1)  # tmp1 = Ax
        self.tmp2 = self.M_il * self.tmp1
        self.A.mult(self.tmp2, self.tmp1)
        y.set_local(self.tmp1.get_local())
        y.apply("insert")

    def approx_root_inv_action(self, x, y):
        """
        L^-1 M_lump^1/2
        Used as a GHEP preconditioner
        eigendecomposition_transformation.tex
        """

        if not isinstance(x, PETScVector):
            x_tmp = PETScVector(x)
        else:
            x_tmp = x

        if not isinstance(y, PETScVector):
            y_tmp = PETScVector(y)
        else:
            y_tmp = y

        self.tmp1 = self.M_rl * x_tmp
        self.A_solver.solve(y_tmp, self.tmp1)


class LumpedPC:
    """
    A preconditioner using the lumped-mass approximation to the
    inverse root of the prior hessian
    See laplacian.approx_root_inv_action
    """
    def __init__(self, lap):
        self.laplacian = lap
        self.action = self.laplacian.approx_root_inv_action

    def setUp(self, pc):
        pass

    @count_calls(1, 'LumpedPC')
    def apply(self, pc, x, y):
        self.action(x, y)

class LaplacianPC:
    """
    A preconditioner using the laplacian inverse_action

    i.e. B^-1  =  L^-1 M L^-1
    """
    def __init__(self, lap):
        self.laplacian = lap
        self.action = self.laplacian.inv_action
        self.x_tmp = Function(self.laplacian.space).vector()
        self.y_tmp = Function(self.laplacian.space).vector()

    def setUp(self, pc):
        pass

    @count_calls(1, 'LaplacianPC')
    def apply(self, pc, x, y):

        self.x_tmp.set_local(x.array)
        self.x_tmp.apply("insert")

        self.action(self.x_tmp, self.y_tmp)

        y.array = self.y_tmp.get_local()
        # TODO - do we need a y.assemble() here?
        # or y.assemblyBegin(), assemblyEnd()?
