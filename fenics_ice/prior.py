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
    """
    Laplacian prior implementation

    NB: although Laplacian makes use of params.inversion.,
    delta_alpha etc come from the *solver* object, because
    these are mutable (e.g. misfit-only hessian)
    """
    def __init__(self, slvr, space):

        self.space = space
        assert space.ufl_element().value_size() in [1, 2]
        self.mixed_space = space.ufl_element().value_size() == 2

        self.delta_alpha = slvr.delta_alpha
        self.delta_beta = slvr.delta_beta
        self.gamma_alpha = slvr.gamma_alpha
        self.gamma_beta = slvr.gamma_beta

        self.alpha_active = slvr.params.inversion.alpha_active
        self.beta_active = slvr.params.inversion.beta_active

        self.test = TestFunctions(space)
        self.trial = TrialFunctions(space)

        # Mass term
        # Construct 1 (scalar function space) or 2 (vector mixed space)
        var_m = [inner(test, trial)*dx for test, trial in zip(self.test, self.trial)]
        self.var_m = var_m

        # Build the form, operators & solvers
        self.construct_mass_operator()
        self.construct_prior_form()
        self.construct_prior_operator()

    def construct_mass_operator(self):
        """Construct the mass operator self.M and its solver self.M_solver"""
        self.M = assemble(sum(self.var_m))

        self.M_solver = KrylovSolver("cg", "sor")
        self.M_solver.parameters.update({"absolute_tolerance": 1.0e-32,
                                         "relative_tolerance": 1.0e-14})
        self.M_solver.set_operator(self.M)

    def construct_prior_operator(self):
        """Construct the prior operator self.A and its solver self.A_solver"""
        self.A = assemble(self.A_form)
        self.A_solver = KrylovSolver("cg", "sor")
        self.A_solver.parameters.update({"absolute_tolerance": 1.0e-32,
                                         "relative_tolerance": 1.0e-14})
        self.A_solver.set_operator(self.A)

        # self.tmp1, self.tmp2 = Function(self.space), Function(self.space)

        self.tmp1, self.tmp2 = Vector(), Vector()
        self.A.init_vector(self.tmp1, 0)
        self.A.init_vector(self.tmp2, 1)

    def construct_prior_form(self):
        """
        Define the form of the prior.

        This is a laplacian.
        """
        # Curvature term
        # Construct 1 (scalar function space) or 2 (vector mixed space)
        var_n = [inner(grad(test), grad(trial))*dx for
                 test, trial in zip(self.test, self.trial)]
        self.var_n = var_n

        var_m = self.var_m

        if self.mixed_space:
            self.alpha_form = self.delta_alpha * var_m[0] + self.gamma_alpha * var_n[0]
            self.beta_form = self.delta_beta * var_m[1] + self.gamma_beta * var_n[1]
            self.A_form = self.alpha_form + self.beta_form
        else:
            if self.alpha_active:
                self.alpha_form = self.delta_alpha * var_m[0] + self.gamma_alpha * var_n[0]
                self.A_form = self.alpha_form
            else:
                self.beta_form = self.delta_beta * var_m[0] + self.gamma_beta * var_n[0]
                self.A_form = self.beta_form

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

    def J_reg(self, alpha, beta, beta_bgd):
        """
        Compute the regularisation term of the cost function

        Returns a list of 1 or 2 terms depending on inversion type.
        """
        assert alpha is not None or beta is not None
        assert (beta is not None) == (beta_bgd is not None)
        assert not self.mixed_space

        space = self.space
        result = [None, None]

        if self.alpha_active:

            alpha_idx = 0
            test = self.test[alpha_idx]
            trial = self.trial[alpha_idx]

            f_alpha = Function(space, name='f_alpha')
            L = ufl.replace(self.alpha_form, {trial: alpha})
            a = test * trial * dx

            solve(a == L, f_alpha)
            J_reg_alpha = self.my_norm(f_alpha)
            result[0] = J_reg_alpha

        if self.beta_active:

            beta_diff = beta-beta_bgd
            beta_idx = 1 if self.mixed_space else 0
            test = self.test[beta_idx]
            # trial = self.trial[beta_idx]

            f_beta = Function(space, name='f_beta')

            # TODO - how to get just a single definition?
            L = ((self.delta_beta * inner(test, beta_diff)) +
                 self.gamma_beta * inner(grad(test), grad(beta))) * dx

            # L = ufl.replace(self.beta_form, {trial: beta})
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
