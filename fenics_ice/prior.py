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
import ufl

class Laplacian(object):

    def __init__(self, params, space):

        invparam = params.inversion

        self.space = space
        self.vdim = space.ufl_element().value_size()

        assert self.vdim in [1, 2]
        assert (self.vdim == 2) == invparam.dual

        self.delta_alpha = invparam.delta_alpha
        self.delta_beta = invparam.delta_beta
        self.gamma_alpha = invparam.gamma_alpha
        self.gamma_beta = invparam.gamma_beta

        self.alpha_active = invparam.alpha_active
        self.beta_active = invparam.beta_active

        self.test = TestFunctions(space)
        self.trial = TrialFunctions(space)

        # Mass and curvature terms
        if self.vdim == 1:
            test = self.test[0]
            trial = self.trial[0]
            var_m = inner(test, trial) * dx
            var_n = inner(grad(test), grad(trial)) * dx
            self.M = assemble(var_m)

        else:
            test_0, test_1 = self.test
            trial_0, trial_1 = self.trial
            var_m[0] = inner(test_0, trial_0) * dx
            var_m[1] = inner(test_1, trial_1) * dx
            var_n[0] = inner(grad(test_0), grad(trial_0)) * dx
            var_n[1] = inner(grad(test_1), grad(trial_1)) * dx
            self.M = assemble(var_m_1 + var_m_2)


        self.M_solver = KrylovSolver("cg", "sor")
        self.M_solver.parameters.update({"absolute_tolerance": 1.0e-32,
                                         "relative_tolerance": 1.0e-14})
        self.M_solver.set_operator(self.M)

        #################################
        # Definition of prior form here!
        #################################
        self.alpha_form = self.delta_alpha * var_m + self.gamma_alpha * var_n
        self.beta_form = self.delta_beta * var_m + self.gamma_beta * var_n

        if self.vdim == 1:
            if invparam.alpha_active:
                self.A_form = self.alpha_form
            else:
                self.A_form = self.beta_form
        else:
            self.A_form = self.alpha_form + self.beta_form

        self.A = assemble(self.A_form)
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

    def my_norm(self, fun):
        """Return the 0.5 inner product of the reg term"""
        return 0.5 * inner(fun, fun)*dx

    def J_reg(self, alpha, beta):
        """
        Compute the regularisation term of the cost function

        Returns a list of 1 or 2 terms depending on inversion type.
        """

        assert alpha is not None or beta is not None
        space = self.space

        result = [None, None]

        if self.alpha_active:

            test = self.test[0]
            trial = self.trial[0]

            f_alpha = Function(space, name='f_alpha')
            L = ufl.replace(self.alpha_form, {trial: alpha})
            a = test * trial * dx

            solve(a == L, f_alpha)
            J_reg_alpha = self.my_norm(f_alpha)
            result[0] = J_reg_alpha

        if self.beta_active:

            test = self.test[1]
            trial = self.trial[1]

            f_beta = Function(space, name='f_beta')
            L = ufl.replace(self.beta_form, {trial: beta})
            a = test * trial * dx

            solve(a == L, f_beta)
            J_reg_beta = self.my_norm(f_beta)
            result[1] = J_reg_beta

        return result

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
