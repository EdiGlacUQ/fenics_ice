# For fenics_ice copyright information see ACKNOWLEDGEMENTS in the fenics_ice
# root directory

# This file is part of fenics_ice.
#
# fenics_ice is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# fenics_ice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from dolfin import *
from tlm_adjoint import *
from .decorators import count_calls, timer

class laplacian(object):

    def __init__(self, params, space):

        invparam = params.inversion

        self.space = space
        self.vdim = space.ufl_element().value_size()

        assert self.vdim in [1, 2]
        assert (self.vdim == 2) == invparam.dual

        if self.vdim == 1:
            test, trial = TestFunction(space), TrialFunction(space)
            var_m = inner(test, trial) * dx
            var_n = inner(grad(test), grad(trial)) * dx
            self.M = assemble(var_m)

        else:
            test_1, test_2 = TestFunctions(space)
            trial_1, trial_2 = TrialFunctions(space)
            var_m_1 = inner(test_1, trial_1) * dx
            var_m_2 = inner(test_2, trial_2) * dx
            var_n_1 = inner(grad(test_1), grad(trial_1)) * dx
            var_n_2 = inner(grad(test_2), grad(trial_2)) * dx
            self.M = assemble(var_m_1 + var_m_2)


        self.M_solver = KrylovSolver("cg", "sor")
        self.M_solver.parameters.update({"absolute_tolerance": 1.0e-32,
                                         "relative_tolerance": 1.0e-14})
        self.M_solver.set_operator(self.M)

        if self.vdim == 1:
            if invparam.alpha_active:
                delta = invparam.delta_alpha
                gamma = invparam.gamma_alpha
            else:
                delta = invparam.delta_beta
                gamma = invparam.gamma_beta
            self.A = assemble(delta * var_m + gamma * var_n)
        else:
            self.A = assemble((invparam.delta_alpha * var_m_1) +
                              (invparam.delta_beta * var_m_2) +
                              (invparam.gamma_alpha * var_n_1) +
                              (invparam.gamma_beta * var_n_2))

        self.A_solver = KrylovSolver("cg", "sor")
        self.A_solver.parameters.update({"absolute_tolerance": 1.0e-32,
                                         "relative_tolerance": 1.0e-14})
        self.A_solver.set_operator(self.A)

        self.tmp1, self.tmp2 = Function(space), Function(space)

        self.tmp1, self.tmp2 = Vector(), Vector()
        self.A.init_vector(self.tmp1, 0)
        self.A.init_vector(self.tmp2, 1)

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
