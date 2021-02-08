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
import ufl
from .decorators import count_calls, timer, flag_errors
from abc import ABC, abstractmethod

class Prior(ABC):
    """Abstraction for prior used by both comp_J_inv and run_eigendec.py"""

    @abstractmethod
    def prior_form(self):
        """Define the prior form"""
        pass

    @abstractmethod
    def action(self, x, y):
        """The action of the prior on a vector"""
        pass

    @abstractmethod
    def inv_action(self, x, y):
        """The inverse action of the prior on a vector"""
        pass

    def __init__(self, slvr, space):
        """Create object members & construct the mass & prior operators"""
        self.solver = slvr
        self.space = space
        assert space.ufl_element().value_size() in [1, 2]
        self.mixed_space = space.ufl_element().value_size() == 2

        self.form_map = {}
        self.LUT = {}

        # Which components?
        self.alpha_idx = 0
        self.beta_idx = 1 if self.mixed_space else 0

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
        var_m = [inner(test, trial) for test, trial in zip(self.test, self.trial)]
        self.var_m = var_m

        # Build the form, operators & solvers
        self.prior_form()
        self.construct_mass_operator()
        self.construct_prior_operator()

    def placeholder_fn(self, name, idx):
        """
        Generate placeholder functions

        These are also added to self.form_map to create the preconditioner form

        And to LUT for the reg form?
        """
        # temporary so choice of space doesn't really matter
        tmp_space = self.space.sub(0) if self.mixed_space else self.space
        tmp = ufl.classes.Coefficient(tmp_space, count=Constant(0).count())

        self.LUT[tmp] = name
        self.form_map[tmp] = self.trial[idx]
        return tmp

    def construct_mass_operator(self):
        """Construct the mass operator self.M and its solver self.M_solver"""
        self.M = assemble(sum(self.var_m) * dx)

        self.M_solver = KrylovSolver("cg", "sor")
        self.M_solver.parameters.update({"absolute_tolerance": 1.0e-32,
                                         "relative_tolerance": 1.0e-14})
        self.M_solver.set_operator(self.M)

    def construct_prior_operator(self):
        """
        Construct the prior operator (self.A) and its solver (self.A_solver)

        Used in the eigendecomp phase as a preconditioner for the misfit Hessian.

        Each Prior implementation must define self.alpha_form and self.alpha_form_map,
        usually in self.prior_form
        """
        assert self.form_map  # check this has been created
        if self.alpha_active:
            alpha_form = ufl.replace(self.alpha_form, self.form_map)

        if self.beta_active:
            beta_form = ufl.replace(self.beta_form, self.form_map)

        if self.mixed_space:
            self.A_form = alpha_form + beta_form
        else:
            if self.alpha_active:
                self.A_form = alpha_form
            else:
                self.A_form = beta_form

        self.A = assemble(self.A_form)
        self.A_solver = KrylovSolver("cg", "sor")
        self.A_solver.parameters.update({"absolute_tolerance": 1.0e-32,
                                         "relative_tolerance": 1.0e-14})
        self.A_solver.set_operator(self.A)

        # self.tmp1, self.tmp2 = Function(self.space), Function(self.space)
        self.tmp1, self.tmp2 = Vector(), Vector()
        self.A.init_vector(self.tmp1, 0)
        self.A.init_vector(self.tmp2, 1)

    def J_reg(self, **kwargs):
        """
        Compute the regularisation term of the cost function

        Returns a list of 1 or 2 terms depending on inversion type.
        """
        # Check we received the args we expected based on prior_form
        assert list(kwargs.keys()) == list(self.LUT.values()), \
            f"Expected kwargs: {self.LUT.values()}"

        mappy = {}
        for k in self.LUT:
            mappy[k] = kwargs[self.LUT[k]]

        assert not self.mixed_space

        space = self.space
        result = [None, None]

        # Note - because this is never used in mixed space mode,
        # no need to mess with alpha_idx, beta_idx
        trial = self.trial[0]
        test = self.test[0]

        if self.alpha_active:

            f_alpha = Function(space, name='f_alpha')
            L = ufl.replace(self.alpha_form, mappy)

            a = test * trial * dx

            # alpha form is negative laplacian
            solve(a == L, f_alpha,  # M^{-1} L alpha
                  solver_parameters={"linear_solver": "direct"})

            J_reg_alpha = self.norm_sq(f_alpha)  # L M^{-1} M M^{-1} L alpha
                                                 # = L M^{-1} L alpha
            result[0] = J_reg_alpha

        if self.beta_active:

            f_beta = Function(space, name='f_beta')

            L = ufl.replace(self.beta_form, mappy)
            a = test * trial * dx

            solve(a == L, f_beta, solver_parameters={"linear_solver": "direct"})
            J_reg_beta = self.norm_sq(f_beta)
            result[1] = J_reg_beta

        return result


class Laplacian(Prior):
    """
    Laplacian prior implementation

    NB: although Laplacian makes use of params.inversion.,
    delta_alpha etc come from the *solver* object, because
    these are mutable (e.g. misfit-only hessian)
    """

    def prior_form(self):
        """
        Define the form of the prior.

        This is a scaled/negative laplacian.
        """

        # These are just *placeholders* which will be substituted before use
        alpha = self.placeholder_fn('alpha', self.alpha_idx)
        beta = self.placeholder_fn('beta', self.beta_idx)
        beta_diff = self.placeholder_fn('beta_diff', self.beta_idx)

        if self.alpha_active:
            self.alpha_form = (self.delta_alpha * inner(alpha,
                                                        self.test[self.alpha_idx]) +
                               self.gamma_alpha * inner(grad(alpha),
                                                        grad(self.test[self.alpha_idx]))) * dx

        if self.beta_active:
            self.beta_form = (self.delta_beta * inner(beta_diff,
                                                      self.test[self.beta_idx]) +
                              self.gamma_beta * inner(grad(beta),
                                                      grad(self.test[self.beta_idx]))) * dx

        # The square norm of the prior for J_reg
        self.norm_sq = lambda x: 0.5 * inner(x, x) * dx

    def action(self, x, y):
        """LM^-1L"""
        self.A.mult(x, self.tmp1)  # tmp1 = Ax
        self.M_solver.solve(self.tmp2, self.tmp1)  # Atmp2 = tmp1
        self.A.mult(self.tmp2, self.tmp1)
        y.set_local(self.tmp1.get_local())
        y.apply("insert")

    def inv_action(self, x, y):
        """L^-1 M L^-1"""
        self.A_solver.solve(self.tmp1, x)
        self.M.mult(self.tmp1, self.tmp2)
        self.A_solver.solve(self.tmp1, self.tmp2)

        y.set_local(self.tmp1.get_local())
        y.apply("insert")


class Laplacian_flt(Laplacian):
    """
    Laplacian prior implementation

    NB: although Laplacian makes use of params.inversion.,
    delta_alpha etc come from the *solver* object, because
    these are mutable (e.g. misfit-only hessian)
    """

    def __init__(self, slvr, space):
        """Get flotation condition & delta_beta_gnd"""
        self.fl_ex = slvr.float_conditional(slvr.H)
        self.delta_beta_gnd = slvr.delta_beta_gnd

        super().__init__(slvr, space)

    def prior_form(self):
        """
        Define the form of the prior.

        This is a scaled/negative laplacian with separate delta_beta
        for floating/grounded regions.
        """
        # These are just *placeholders* which will be substituted before use
        alpha = self.placeholder_fn('alpha', self.alpha_idx)
        beta = self.placeholder_fn('beta', self.beta_idx)
        beta_diff = self.placeholder_fn('beta_diff', self.beta_idx)
        fl_ex = self.fl_ex

        if self.alpha_active:
            self.alpha_form = (self.delta_alpha * inner(alpha,
                                                        self.test[self.alpha_idx]) +
                               self.gamma_alpha * inner(grad(alpha),
                                                        grad(self.test[self.alpha_idx]))) * dx

        if self.beta_active:
            self.beta_form = ((self.delta_beta * fl_ex *
                              inner(beta_diff, self.test[self.beta_idx])) +

                              (self.delta_beta_gnd * (1.0 - fl_ex) *
                              inner(beta_diff, self.test[self.beta_idx])) +

                              self.gamma_beta * inner(grad(beta),
                                                      grad(self.test[self.beta_idx]))) * dx

        # The square norm of the prior for J_reg
        self.norm_sq = lambda x: 0.5 * inner(x, x) * dx


class LaplacianPC:
    """
    A preconditioner using the laplacian inverse_action

    i.e. B^-1  =  L^-1 M L^-1
    """

    @flag_errors
    def __init__(self, lap):
        self.laplacian = lap
        self.action = self.laplacian.inv_action
        self.x_tmp = Function(self.laplacian.space).vector()
        self.y_tmp = Function(self.laplacian.space).vector()

    @flag_errors
    def setUp(self, pc):
        pass

    @count_calls(1, 'LaplacianPC')
    @flag_errors
    def apply(self, pc, x, y):

        self.x_tmp.set_local(x.array)
        self.x_tmp.apply("insert")

        self.action(self.x_tmp, self.y_tmp)

        y.array = self.y_tmp.get_local()
        # TODO - do we need a y.assemble() here?
        # or y.assemblyBegin(), assemblyEnd()?
