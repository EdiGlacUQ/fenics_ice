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

import time
from pathlib import Path
import numpy as np

from fenics import *
from fenics_ice import inout, prior
from tlm_adjoint.fenics import *
from tlm_adjoint.hessian_optimization import *

from tlm_adjoint.fenics.backend_code_generator_interface import matrix_multiply

from .minimize_l_bfgs import minimize_l_bfgs
from .minimize_l_bfgs import \
    line_search_rank0_scipy_scalar_search_wolfe1 as line_search_wolfe1
from .minimize_l_bfgs import \
    line_search_rank0_scipy_scalar_search_wolfe2 as line_search_wolfe2


#from dolfin_adjoint import *
#from dolfin_adjoint_custom import EquationSolver
import ufl
import logging

log = logging.getLogger("fenics_ice")

class ssa_solver:

    def __init__(self, model, mixed_space=False):

        # Enable aggressive compiler options
        parameters["form_compiler"]["optimize"] = False
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math -march=native"
        parameters["form_compiler"]["precision"] = 16

        self.model = model
        self.model.solvers.append(self)
        self.params = model.params
        self.mixed_space = mixed_space

        # Mesh/Function Spaces
        self.mesh = model.mesh
        self.V = model.V
        self.Q = model.Q
        self.Qp = model.Qp
        self.M = model.M
        self.RT = model.RT

        # Fields
        self.bed = model.bed
        self.H_np = model.H_np
        self.H = model.H
        self.beta_bgd = model.beta_bgd
        self.bmelt = model.bmelt
        self.smb = model.smb
        self.latbc = model.latbc

        self.set_inv_params()

        self.gen_control_fns()
        # Note - *copying* controls from model
        self.set_control_fns([model.alpha, model.beta], initial=True)

        # self.test_outfile = None
        # self.f_alpha_file = None

        # Parameterization of alpha/beta
        self.bglen_to_beta = model.bglen_to_beta
        self.beta_to_bglen = model.beta_to_bglen

        # Facet normals
        self.nm = model.nm

        # Save observations for inversions
        try:
            self.u_obs = model.u_obs
            self.v_obs = model.v_obs
            self.u_std = model.u_std
            self.v_std = model.v_std
            self.uv_obs_pts = model.uv_obs_pts
        except:
            pass

        # Trial/Test Functions
        self.U = Function(self.V, name="U")
        self.U_np = Function(self.V, name="U_np")
        self.Phi = TestFunction(self.V)
        self.Ksi = TestFunction(self.M)
        self.pTau = TestFunction(self.Qp)

        self.trial_H = TrialFunction(self.M)

        # Facets
        self.ff = model.ff

        # Measures
        self.dx = Measure('dx', domain=self.mesh)
        self.dS = Measure('dS', domain=self.mesh)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.ff)

        self.dIce = self.dx  # just an alias
        self.dt = Constant(self.params.time.dt, name="dt")

        self.eigenvals = None
        self.eigenfuncs = None

    def set_inv_params(self):

        invparam = self.params.inversion
        self.delta_alpha = invparam.delta_alpha
        self.gamma_alpha = invparam.gamma_alpha
        self.delta_beta = invparam.delta_beta
        self.delta_beta_gnd = invparam.delta_beta_gnd
        self.gamma_beta = invparam.gamma_beta

    def zero_inv_params(self):

        self.delta_alpha = 1E-10
        self.gamma_alpha = 1E-10
        self.delta_beta = 1E-10
        self.gamma_beta = 1E-10

        if self.delta_beta_gnd is not None:
            self.delta_beta_gnd = 1E-10

    def get_mixed_space(self):
        """Return the mixed function space for alphaXbeta"""
        el = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        mixedElem = el * el
        if not self.params.mesh.periodic_bc:
            return FunctionSpace(self.mesh,
                                 mixedElem)
        else:
            mesh_length = fice_mesh.get_mesh_length(self.mesh)
            return FunctionSpace(self.mesh,
                                 mixedElem,
                                 constrained_domain=PeriodicBoundary(mesh_length))

    def gen_control_fns(self):
        """
        Set up the space & control function(s)

        If solver.mixed_space = True, this creates a CGxCG space and variable
        self._alphaXbeta which holds both controls. This is used in e.g. dual
        eigendecomposition.

        If solver.mixed_space = False, no mixed space is created, and self._alpha,
        self._beta live in the space self.Qp
        """

        if self.mixed_space:
            assert self.params.inversion.dual  # only need mixed space for 2 controls
            self.QQ = self.get_mixed_space()
            self._alphaXbeta = Function(self.QQ, name="alphaXbeta", static=True)
            self._cntrl_assigner = FunctionAssigner(self.QQ, [self.Qp]*2)

            self._alpha = None
            self._beta = None
        else:
            self._alpha = Function(self.Qp, name='alpha', static=True)
            self._beta = Function(self.Qp, name='beta', static=True)

            self.QQ = None
            self._alphaXbeta = None

    def set_control_fns(self, f, initial=False):
        """
        Set (or assign) the control functions

        If initial=True, function values are assigned, else functions themselves
        are set. The option to *assign* is only required because, in mixed space
        case, we specify mdl.alpha and mdl.beta, so must make use of FunctionAssigner.
        For consistency, 'initial' always assigns even when not using mixed space.

        """

        if not isinstance(f, (list, tuple)):
            f = [f]

        invconfig = self.params.inversion
        if self.mixed_space:  # operate on _alphaXbeta

            assert invconfig.dual
            if initial:
                assert len(f) == 2
                self._cntrl_assigner.assign(self._alphaXbeta, f)
            else:
                assert len(f) == 1
                self._alphaXbeta = f[0]

        else:  # operate on _alpha, _beta

            if initial:
                assert len(f) == 2
                self._alpha.assign(f[0])
                self._beta.assign(f[1])
            elif invconfig.dual:
                assert len(f) == 2
                self._alpha = f[0]
                self._beta = f[1]
            else:
                assert len(f) == 1
                if invconfig.alpha_active:
                    self._alpha = f[0]
                else:
                    self._beta = f[0]

    def update_model_fns(self):
        """
        Update model's control variables from solver

        Note alpha_proj returns an unannotated copy of the solver's controls.
        """
        mdl = self.model
        invconfig = self.params.inversion
        if invconfig.alpha_active:
            mdl.alpha = self.alpha_proj
        if invconfig.beta_active:
            mdl.beta = self.beta_proj

    @property
    def alpha(self):
        """
        Get alpha function

        The returned function depends on:
        - self.mixed_space - _alpha, or component of _alphaXbeta?
        - alpha_active - return the function itself or a copy/projection?
        """
        if self.mixed_space:
            return self._alphaXbeta[0]
        else:
            return self._alpha

    @property
    def beta(self):
        """
        Get beta function

        The returned function depends on:
        - self.mixed_space - _beta, or component of _alphaXbeta?
        - beta_active - return the function itself or a copy/projection?
        """
        if self.mixed_space:
            return self._alphaXbeta[1]
        else:
            return self._beta

    @property
    def alpha_proj(self):
        """Return a copy of solver's alpha, guaranteed in Qp"""
        if(self.mixed_space):
            alpha = project(self._alphaXbeta[0], self.Qp)
        else:
            alpha = self._alpha.copy(deepcopy=True)

        alpha.rename("alpha", "")
        return alpha

    @property
    def beta_proj(self):
        """Return a copy of solver's beta, guaranteed in Qp"""
        if(self.mixed_space):
            beta = project(self._alphaXbeta[1], self.Qp)
        else:
            beta = self._beta.copy(deepcopy=True)
        beta.rename("beta", "")
        return beta

    def get_qoi_func(self):
        qoi_dict = {'vaf': self.comp_Q_vaf,
                    'h2': self.comp_Q_h2}
        choice = self.params.error_prop.qoi
        return qoi_dict[choice.lower()]  # flexible case

    def def_mom_eq(self):
        """Define the momentum equation to be solved in solve_mom_eq"""

        # Simplify accessing fields and parameters
        constants = self.params.constants
        bed = self.bed
        H = self.H
        alpha = self.alpha

        rhoi = constants.rhoi
        rhow = constants.rhow
        delta = 1.0 - rhoi/rhow
        g = constants.g
        n = constants.glen_n
        tol = constants.float_eps
        dIce = self.dIce
        ds = self.ds
        sl = self.params.ice_dynamics.sliding_law
        vel_rp = constants.vel_rp

        # Vector components of trial function
        u, v = split(self.U)

        # Vector components of test function
        Phi = self.Phi
        Phi_x, Phi_y = split(Phi)

        # Derivatives
        u_x, u_y = u.dx(0), u.dx(1)
        v_x, v_y = v.dx(0), v.dx(1)

        # Viscosity
        # If nu varies by 6-7 orders of magnitude, taylor verification likely to fail
        U_marker = Function(self.U.function_space(), name="%s_marker" % self.U.name())
        nu = self.viscosity(U_marker)

        # Switch parameters
        H_flt = -rhow/rhoi * bed
        fl_ex = self.float_conditional(H, H_flt)
        B2 = self.sliding_law(alpha, U_marker)

        ##############################################
        # Construct weak form
        ##############################################

        # Driving stress quantities
        F = (1 - fl_ex) * 0.5*rhoi*g*H**2 + \
            (fl_ex) * 0.5*rhoi*g*(delta*H**2 + (1-delta)*H_flt**2 )      # order mag ~ 1E10

        W = (1 - fl_ex) * rhoi*g*H + \
            (fl_ex) * rhoi*g*H_flt                                       # order mag ~ 1E7

        # Depth of the submarine portion of the calving front
        # ufl.Max to avoid edge case of land termininating calving front
        draft = ufl.Max(((fl_ex) * (rhoi / rhow) * H      # order mag ~ 1E3
                 - (1 - fl_ex) * bed), Constant(0.0, name="Const No Draft"))

        # Terminating margin boundary condition
        # The -F term is related to the integration by parts in the weak form
        sigma_n = 0.5 * rhoi * g * ((H ** 2) - (rhow / rhoi) * (draft ** 2))   # order mag ~ 1E10

        self.mom_F = (
            # Membrance Stresses
            -inner(grad(Phi_x), H * nu * as_vector([4 * u_x + 2 * v_y, u_y + v_x]))  # nu ~ 1e8, H ~ 1e3, u_x ~ 1e-3
            * self.dIce

            - inner(grad(Phi_y), H * nu * as_vector([u_y + v_x, 4 * v_y + 2 * u_x]))
            * self.dIce

            # Basal Drag
            - inner(Phi, (1.0 - fl_ex) * B2 * as_vector([u, v]))   # B2 spans 1e-7, 1e7,  2nd step u ~ 1e-18 to 1e2
            * self.dIce

            # Driving Stress
            + ( div(Phi)*F - inner(grad(bed), W*Phi) )
            * self.dIce
        )

        # Natural BC term which falls out of weak form:
        # Doesn't apply when domain is periodic
        if not self.params.mesh.periodic_bc:
            self.mom_F -= inner(Phi * F, self.nm) * ds

        ##########################################
        # Boundary Conditions
        ##########################################

        # Dirichlet
        self.flow_bcs = []
        for bc in self.params.bcs:
            if bc.flow_bc == "obs_vel":
                dirichlet_condition = self.latbc
            elif bc.flow_bc == "no_slip":
                dirichlet_condition = Constant((0.0, 0.0))
            elif bc.flow_bc == "free_slip":
                raise NotImplementedError
            else:
                continue

            # Add the dirichlet condition to list
            self.flow_bcs.extend([DirichletBC(self.V,
                                              dirichlet_condition,
                                              self.ff,
                                              lab) for lab in bc.labels])

        # Neumann
        # We construct a MeasureSum of different exterior facet sections
        # (for now this is only for the calving front)
        # Then we add to the neumann_bcs list:  calving_weak_form_bc * ds(calving_fronts)
        self.neumann_bcs = []
        for bc in self.params.bcs:
            if bc.flow_bc == "calving":
                condition = inner(Phi * sigma_n, self.nm)
            else:  # don't need to do anything for 'natural'
                continue

            measure_list = [ds(i) for i in bc.labels]
            measure_sum = measure_list[0]
            for m in measure_list[1:]:
                measure_sum += m

            self.neumann_bcs.append(condition * measure_sum)

        # Add the Neumann BCs to the weak form
        for neumann in self.neumann_bcs:
            self.mom_F += neumann

        ###########################################
        # Expand derivatives & add marker functions
        ###########################################
        self.mom_Jac_p = ufl.algorithms.expand_derivatives(
            ufl.replace(derivative(self.mom_F, self.U), {U_marker: self.U}))

        self.mom_F = ufl.algorithms.expand_derivatives(
            ufl.replace(self.mom_F, {U_marker: self.U}))

        self.mom_Jac = ufl.algorithms.expand_derivatives(
            derivative(self.mom_F, self.U))

    def sliding_law(self, alpha, U):

        constants = self.params.constants

        bed = self.bed
        H = self.H
        rhoi = constants.rhoi
        rhow = constants.rhow
        g = constants.g
        sl = self.params.ice_dynamics.sliding_law
        vel_rp = constants.vel_rp

        fl_ex = self.float_conditional(H)

        C = alpha*alpha
        u, v = split(U)

        if sl == 'linear':
            B2 = C

        elif sl == 'budd':
            N = (1-fl_ex)*(H*rhoi*g + ufl.Min(bed, 0.0)*rhow*g)
            U_mag = sqrt(U[0]**2 + U[1]**2 + vel_rp**2)
            # Need to catch N <= 0.0 here, as it's raised to
            # 1/3 (forward) and -2/3 (adjoint)
            N_term = ufl.conditional(N > 0.0, N ** (1.0/3.0), 0)
            B2 = (1-fl_ex)*(C * N_term * U_mag**(-2.0/3.0))

        return B2

    def solve_mom_eq(self, annotate_flag=None):
        """Solve the momentum equation defined in def_mom_eq"""

        t0 = time.time()

        newton_params = self.params.momsolve.newton_params
        picard_params = self.params.momsolve.picard_params
        J_p = self.mom_Jac_p

        momsolver = MomentumSolver(self.mom_F == 0,
                                   self.U,
                                   bcs=self.flow_bcs,
                                   J_p=J_p,
                                   picard_params=picard_params,
                                   solver_parameters=newton_params)

        momsolver.solve(annotate=annotate_flag)

        t1 = time.time()
        info("Time for solve: {0}".format(t1-t0))

    def def_thickadv_eq(self):
        U_np = self.U_np
        Ksi = self.Ksi
        trial_H = self.trial_H
        H_np = self.H_np
        H = self.H
        H_init = self.H_init
        bmelt = self.bmelt
        smb = self.smb
        dt = self.dt
        nm = self.nm
        dIce = self.dIce
        ds = self.ds
        dS = self.dS

        fl_ex = self.float_conditional(H)

        # Notes on use of DG(0) here:
        # ==========================
        # Thickness here is analogous to pressure in incompressible stokes system.
        # Equal order elements result in pressure modes (wiggles, inf-sup stability, LBB)
        # So 'thickness modes' will appear unless we use DG(0)

        # Crank Nicholson

        # self.thickadv = (inner(Ksi, ((trial_H - H_np) / dt)) * dIce
        # - inner(grad(Ksi), U_np * 0.5 * (trial_H + H_np)) * dIce
        # + inner(jump(Ksi), jump(0.5 * (dot(U_np, nm) + abs(dot(U_np, nm)))
        # * 0.5 * (trial_H + H_np))) * dS
        # + conditional(dot(U_np, nm) > 0, 1.0, 0.0)
        # *inner(Ksi, dot(U_np * 0.5 * (trial_H + H_np), nm))*ds # Outflow
        # + conditional(dot(U_np, nm) < 0, 1.0 , 0.0)
        # *inner(Ksi, dot(U_np * H_init, nm))*ds # Inflow
        # + bmelt*Ksi*dIce_flt) #basal melting

        # Backward Euler
        self.thickadv = (
            # dH/dt
            + inner(Ksi, ((trial_H - H_np) / dt)) * dIce

            # Advection
            - inner(grad(Ksi), U_np * trial_H) * dIce

            # Spatial gradient
            + inner(jump(Ksi), jump(0.5 * (dot(U_np, nm) + abs(dot(U_np, nm))) * trial_H))
            * dS

            # Outflow at boundaries
            + conditional(dot(U_np, nm) > 0, 1.0, 0.0)*inner(Ksi, dot(U_np * trial_H, nm))
            * ds

            # Inflow at boundaries
            + conditional(dot(U_np, nm) < 0, 1.0, 0.0)*inner(Ksi, dot(U_np * H_init, nm))
            * ds

            # basal melting
            + bmelt*Ksi*fl_ex*dIce

            # surface mass balance
            - smb*Ksi*dIce
        )

        # # Forward euler
        # self.thickadv = (inner(Ksi, ((trial_H - H_np) / dt)) * dIce
        # - inner(grad(Ksi), U_np * H_np) * dIce
        # + inner(jump(Ksi), jump(0.5 * (dot(U_np, nm) + abs(dot(U_np, nm))) * H_np)) * dS
        #
        # Outflow
        # + conditional(dot(U_np, nm) > 0, 1.0, 0.0)*inner(Ksi, dot(U_np * H_np, nm))*ds
        #
        # Inflow
        # + conditional(dot(U_np, nm) < 0, 1.0 , 0.0)*inner(Ksi, dot(U_np * H_init, nm))*ds
        # + bmelt*Ksi*dIce_flt) #basal melting

        self.H_bcs = []

    def solve_thickadv_eq(self):

        H = self.H
        a, L = lhs(self.thickadv), rhs(self.thickadv)
        solve(a == L, H, bcs=self.H_bcs,
              solver_parameters={"linear_solver": "lu",
                                 "absolute_tolerance": 1e-10,
                                 "relative_tolerance": 1e-11,
              })  # Not sure these solver params are necessary (linear solve)

    def timestep(self, save=1, adjoint_flag=1, qoi_func=None ):
        """
        Time evolving model
        Returns the QoI
        """

        # Read timestep info
        config = self.params.time
        n_steps = config.total_steps
        dt = config.dt
        run_length = config.run_length

        outdir = self.params.io.output_dir

        t = 0.0

        self.Qval_ts = np.zeros(n_steps+1)
        Q = Functional(name="Q")
        Q_is = []

        U = self.U
        U_np = self.U_np
        # H = self.H
        H_np = self.H_np

        if adjoint_flag:
            num_sens = self.params.time.num_sens
            t_sens = np.flip(np.linspace(run_length, 0, num_sens))

            n_sens = np.round(t_sens/dt)

            reset_manager()
            start_annotating()

            inout.configure_tlm_checkpointing(self.params)

        self.def_thickadv_eq()
        self.def_mom_eq()
        self.solve_mom_eq()
        U_np.assign(U)

        if qoi_func is not None:
            qoi = qoi_func()
            self.Qval_ts[0] = assemble(qoi)

        if adjoint_flag:
            if 0.0 in n_sens:
                Q_i = Functional(name="Q_i")
                Q_i.assign(qoi)
                Q_is.append(Q_i)
                Q.addto(Q_i.fn())

            new_block()

        if save:
            Hfile = Path(outdir) / "_".join((self.params.io.run_name,
                                             'H_ts.xdmf'))
            Ufile = Path(outdir) / "_".join((self.params.io.run_name,
                                             'U_ts.xdmf'))

            xdmf_hts = XDMFFile(self.mesh.mpi_comm(), str(Hfile))
            xdmf_uts = XDMFFile(self.mesh.mpi_comm(), str(Ufile))

            xdmf_hts.write(H_np, 0.0)
            xdmf_uts.write(U_np, 0.0)

        ########################
        # Main timestepping loop
        ########################
        for n in range(n_steps):
            begin("Starting timestep %i of %i, time = %.16e a" % (n + 1, n_steps, t))

            # Solve

            # Simple Scheme
            self.solve_thickadv_eq()
            H_np.assign(self.H)

            self.solve_mom_eq()
            U_np.assign(self.U)

            # increment time
            n += 1
            t = n * float(dt)

            # Record
            if qoi_func is not None:
                qoi = qoi_func()
                self.Qval_ts[n] = assemble(qoi)

                if adjoint_flag:
                    if n in n_sens:
                        Q_i = Functional(name="Q_i")
                        Q_i.assign(qoi)
                        Q_is.append(Q_i)
                        Q.addto(Q_i.fn())

            if n < n_steps and adjoint_flag:
                new_block()

            if save:
                xdmf_hts.write(H_np, t)
                xdmf_uts.write(U_np, t)

        if save:
            xdmf_hts.close()
            xdmf_uts.close()

        return Q_is if qoi_func is not None else None

    # def forward_ts_alpha(self,aa):
    #     clear_caches()
    #     self.timestep()
    #     new_block()
    #     self.Q_vaf = self.comp_Q_vaf()
    #     Q_vaf = Functional()
    #     Q_vaf.assign(self.Q_vaf)
    #     return Q_vaf

    def forward(self, f):
        """
        Runs the forward model w/ controls 'f' and returns cost function 'J'

        Which controls are set (alpha, beta, both) is determined by
        self.set_control_fns() based on params. This function replaces 3
        functions: forward_alpha, forward_beta, forward_dual.

        Note that it is important that self.alpha/beta are *redefined*
        rather than simply assigned to, because of how tlm_adjoint works.
        An annotated connection between 'f' and 'J' must be created, so
        self._alphaXbeta or _alpha, _beta must be set to f.
        """

        clear_caches()

        self.set_control_fns(f)

        # Uncomment to enable alpha output per inversion iteration
        # if not self.test_outfile:
        #     self.test_outfile = File(os.path.join('invoutput_data','alpha_test.pvd'))
        # self.test_outfile << self.alpha

        self.def_mom_eq()
        self.solve_mom_eq()
        J = self.comp_J_inv()
        return J

    def get_control(self):
        """Return the solver's control functions (alpha, beta or alphaXbeta)"""
        invconfig = self.params.inversion
        if self.mixed_space:
            return [self._alphaXbeta]
        else:
            if invconfig.dual:
                return [self.alpha, self.beta]
            elif invconfig.alpha_active:
                return [self.alpha]
            else:
                return [self.beta]

    def get_control_space(self):
        """Return the function space of the control function(s)"""
        if self.mixed_space:
            return self.QQ
        else:
            return self.Qp

    def inversion(self):
        """
        Minimize the cost function J and optimize control functions.

        Runs the annotated forward model and then uses tlm_adjoint
        and L-BFGS for optimisation.

        Also sets the model's control functions to the optimized result.
        """
        config = self.params.inversion

        cntrl = self.get_control()
        forward = self.forward

        if(config.verbose):
            inv_vals = []

        reset_manager()
        clear_caches()

        start_annotating()

        J = forward(cntrl)
        stop_annotating()

        inv_grad_writer = inout.XDMFWriter(inout.gen_path(self.params, 'inv_grads', '.xdmf'),
                                           comm=self.mesh.mpi_comm())

        ##########################################
        # Uncomment for dependency graph output:
        ##########################################
        # from fenics_ice import graphviz
        # manager_graph = graphviz.dot()
        # with open("/home/joe/sources/fenics_ice/debug_mixed.dot", "w") as outfile:
        #     outfile.write(manager_graph)

        ##########################################
        # Uncomment for Taylor Verification:
        ##########################################
        # dJ = compute_gradient(J, self.alpha)
        # ddJ = Hessian(forward)
        # min_order = taylor_test(forward, self.alpha, J_val=J.value(),
        #                         dJ=dJ, ddJ=ddJ, seed=1.0e-6)

        def l_bfgs_converged(it, old_J_val, new_J_val,
                             new_cc, new_dJ, cc_change, dJ_change):
            # Convergence tests and defaults as described in SciPy 1.5.2
            # scipy.optimize._minimize._minimize_lbfgsb documentation

            converged = False
            if(config.verbose):
                info(f"Inversion inner iteration: {it}")

            # Compute functional convergence
            f_criterion = ((old_J_val - new_J_val)
                           / max(abs(old_J_val), abs(new_J_val), 1.0))

            # And test it if requested
            if((config.ftol is not None) and (f_criterion <= config.ftol)):
                info("ftol convergence")
                converged = True

            # Compute gradient convergence
            if config.dual:
                g_criterion = [function_linf_norm(new_dJ[0]), function_linf_norm(new_dJ[1])]
                inv_grad_writer.write(new_dJ[0], name='djdAlpha', step=it)
                inv_grad_writer.write(new_dJ[1], name='djdBeta', step=it)
            elif config.alpha_active:
                g_criterion = [function_linf_norm(new_dJ), 0.0]
                inv_grad_writer.write(new_dJ, name='djdAlpha', step=it)
            else:
                g_criterion = [0.0, function_linf_norm(new_dJ)]
                inv_grad_writer.write(new_dJ, name='djdBeta', step=it)

            # And test it if requested
            if((config.gtol is not None) and (np.max(g_criterion) <= config.gtol)):
                info("gtol convergence")
                converged = True

            if(config.verbose):
                inv_vals.append((new_J_val, f_criterion, *g_criterion))

            if it < config.min_iter:
                converged = False

            return converged

        # L_solver = KrylovSolver(L_mat.copy(), "cg", "sor")
        # L_solver.parameters.update({"relative_tolerance": 1.0e-14,
        #                             "absolute_tolerance": 1.0e-32})

        Qp_test, Qp_trial = TestFunction(self.Qp), TrialFunction(self.Qp)

        M_mat = assemble(inner(Qp_trial, Qp_test) * self.dIce)
        M_solver = KrylovSolver(M_mat.copy(), "cg", "sor")
        M_solver.parameters.update({"relative_tolerance": 1.0e-14,
                                    "absolute_tolerance": 1.0e-32})

        # def B_0(x):
        #     """
        #     Step length something...

        #     2.0 * L M^-1 L
        #     """
        #     L_action = L_mat * x.vector()

        #     M_inv_L_action = function_new(x, name="M_inv_L_action")
        #     M_solver.solve(M_inv_L_action.vector(), L_action)
        #     del L_action

        #     B_0_action = function_new(x, name="B_0_action")
        #     B_0_action.vector().axpy(2.0, L_mat * M_inv_L_action.vector())

        #     return B_0_action

        # TODO - refactor here - these should go in an 'inversion' module
        # or in minimize_l_bfgs...
        def H_M_0(*X):
            """
            Initial guess for inverse hessian action (mass matrix prec)
            M^-1
            """

            M_inv_action = []
            for i, x in enumerate(X):
                this_action = function_new(x, name=f"M_inv_action_{i}")
                M_solver.solve(this_action.vector(), x.vector())
                M_inv_action.append(this_action)

            return M_inv_action

        # def H_0(x):
        #     """
        #     Initial guess for inverse hessian action
        #     0.5 L^-1 M L^-1
        #     """
        #     L_inv_action = function_new(x, name="L_inv_action")
        #     L_solver.solve(L_inv_action.vector(), x.vector().copy())

        #     M_L_inv_action = M_mat * L_inv_action.vector()

        #     H_0_action = function_new(x, name="H_0_action")
        #     L_solver.solve(H_0_action.vector(), M_L_inv_action)
        #     del M_L_inv_action

        #     function_set_values(H_0_action,
        #                         0.5 * function_get_values(H_0_action))

        #     return H_0_action

        def B_M_0(*X):
            """
            Initial guess for hessian action

            M
            """
            B_0_action = []
            for i, x in enumerate(X):
                this_action = function_new(x, name=f"B_0_action_{i}")
                # M_action = M_mat * x.vector()
                M_action = matrix_multiply(M_mat, x.vector())
                this_action.vector()[:] = M_action
                this_action.vector().apply("insert")
                B_0_action.append(this_action)

            return B_0_action

        # L-BFGS-B line search configuration. For parameter values see
        # SciPy
        #     Git master 0a7bc723d105288f4b728305733ed8cb3c8feeb5
        #     June 20 2020
        #     scipy/optimize/lbfgsb_src/lbfgsb.f
        #     lnsrlb subroutine

        # For mass matrix preconditioner (still not working properly),
        # use:
        # c1=1.0e-4, c2=0.9999,
        # H_0=H_M_0, M=B_M_0, M_inv=H_M_0)

        if config.mass_precon:
            H_0_fun = H_M_0
            M_fun = B_M_0
        else:
            H_0_fun = None
            M_fun = None

        cntrl_opt, result = minimize_l_bfgs(
            forward, cntrl, J0=J,
            m=config.m,
            s_atol=config.s_atol,
            g_atol=config.g_atol,
            c1=config.c1, c2=config.c2,
            converged=l_bfgs_converged,
            line_search_rank0=line_search_wolfe1,
            theta_scale=config.theta_scale,
            delta=config.delta_lbfgs,
            line_search_rank0_kwargs={"xtol": config.wolfe_xtol,
                                      "amax": config.wolfe_amax},
            H_0=H_0_fun, M=M_fun, M_inv=H_0_fun,
            block_theta_scale=config.dual,
            max_its=config.max_iter)

        self.set_control_fns(cntrl_opt)

        inv_grad_writer.close()

        # copy control variables back to model
        self.update_model_fns()

        if(config.verbose):
            info(f"Inversion terminated because {result[2]}")
            inout.write_inversion_info(self.params, inv_vals)

        self.def_mom_eq()

        # Re-compute velocities with inversion results
        reset_manager()
        clear_caches()
        start_annotating()
        self.solve_mom_eq()
        stop_annotating()

        # Print out inversion results/parameter values
        self.J_inv = self.comp_J_inv(verbose=True)

    def epsilon(self, U):
        """Return the strain-rate tensor of self.U"""
        epsdot = sym(grad(U))
        return epsdot

    def effective_strain_rate(self, U):
        """Return the effective strain rate squared"""
        eps_rp = self.params.constants.eps_rp

        eps = self.epsilon(U)
        exx = eps[0, 0]
        eyy = eps[1, 1]
        exy = eps[0, 1]

        # Second invariant of the strain rate tensor squared
        eps_2 = (exx**2 + eyy**2 + exx*eyy + (exy)**2 + eps_rp**2)

        return eps_2

    def viscosity(self, U):
        """Compute the viscosity"""
        B = self.beta_to_bglen(self.beta)
        n = self.params.constants.glen_n

        eps_2 = self.effective_strain_rate(U)
        nu = 0.5 * B * eps_2**((1.0-n)/(2.0*n))

        return nu

    def float_conditional(self, H, H_float=None):
        """Compute a ufl Conditional where floating=1, grounded=0"""

        if not self.params.ice_dynamics.allow_flotation:
            return Constant(0.0)

        if H_float is None:
            constants = self.params.constants
            rhow = constants.rhow
            rhoi = constants.rhoi
            H_float = -(rhow/rhoi) * self.bed

        # Note: cell=triangle just suppresses a UFL warning ("missing cell")
        fl_ex = ufl.operators.Conditional(H <= H_float,
                                          Constant(1.0, cell=triangle, name="Const Floating"),
                                          Constant(0.0, cell=triangle, name="Const Grounded"))

        return fl_ex

    def comp_J_inv(self, verbose=False):
        """
        Compute the value of the cost function

        Note: 'verbose' significantly decreases speed
        """
        invconfig = self.params.inversion

        # What are we inverting for?:
        do_alpha = invconfig.alpha_active
        do_beta = invconfig.beta_active

        u, v = split(self.U)

        # Observed velocities
        u_obs = self.u_obs
        v_obs = self.v_obs
        u_std = self.u_std
        v_std = self.v_std
        uv_obs_pts = self.uv_obs_pts

        # Control functions
        alpha = self.alpha
        beta = self.beta
        beta_bgd = self.beta_bgd
        betadiff = beta-beta_bgd

        # Measure
        dIce = self.dIce
        # ds = self.ds
        # nm = self.nm

        # Determine observations within our mesh partition
        # Note that although cell_max counts ghost_cells,
        # compute_first_entity_collision seems to ignore
        # ghost elements, and so arrives at the right answer
        cell_max = self.mesh.cells().shape[0]
        obs_local = np.zeros_like(u_obs, dtype=bool)

        bbox = self.mesh.bounding_box_tree()
        for i in range(uv_obs_pts.shape[0]):
            p = Point(uv_obs_pts[i, 0], uv_obs_pts[i, 1])
            obs_local[i] = bbox.compute_first_entity_collision(p) <= cell_max

        if np.sum(obs_local) == 0:
            raise NotImplementedError("At least one partition has no velocity observations. "
                                      "Need to implement a dummy point w/ semi-inner-product "
                                      "to handle this case.")

        local_cnt = np.sum(obs_local)
        local_obs_pts = [(u, v) for u, v in uv_obs_pts[obs_local]]

        # Sample Discrete Points

        # Arbitrary mesh to define function for interpolated variables

        # Gather info from other partitions on global point counts etc
        comm = self.mesh.mpi_comm()
        rank = comm.rank

        global_vcnts = comm.allgather(local_cnt)  # points on each partition
        global_vcnt = sum(global_vcnts)  # total number of points

        # TODO - better way to check no duplicate points?
        # assert global_vcnt == len(obs_local), ("Mismatch between total number of observation "
        # "points and the sum of those assigned to each partition. "
        # "Note that this might not necessarily be an error - should be handled better.")

        # each partition has 1 fewer elem than points
        global_ecnts = [vc-1 for vc in global_vcnts]
        global_ecnt = sum(global_ecnts)

        # Compute global idx offsets
        vidx_offset = 0
        for i in range(rank):
            vidx_offset += global_vcnts[i]

        obs_mesh = Mesh()

        editor = MeshEditor()
        editor.open(obs_mesh, "interval", 1, 2)
        editor.init_vertices_global(local_cnt, global_vcnt)
        editor.init_cells_global(max(local_cnt-1, 0), global_ecnt)

        # Add vertices
        for i in range(local_cnt):
            editor.add_vertex_global(i, i+vidx_offset, local_obs_pts[i])

        # Add elements  - NOTE - possibly need to add cells to connect partitions
        # Also may need to handle edge case where a partition has no obs pts (caught above)
        for i in range(local_cnt-1):
            editor.add_cell(i, [i, i+1])

        editor.close()
        obs_mesh.init()
        obs_mesh.order()

        obs_space = FunctionSpace(obs_mesh, "CG", 1)

        # NB: this dofmap approach depends on CG space! (dofs at nodes)
        dofmap = vertex_to_dof_map(obs_space)

        u_obs_pts = Function(obs_space, name='u_obs_pts')
        v_obs_pts = Function(obs_space, name='v_obs_pts')
        u_std_pts = Function(obs_space, name='u_std_pts')
        v_std_pts = Function(obs_space, name='v_std_pts')

        u_obs_pts.vector()[:] = u_obs[obs_local][dofmap]
        v_obs_pts.vector()[:] = v_obs[obs_local][dofmap]
        u_std_pts.vector()[:] = u_std[obs_local][dofmap]
        v_std_pts.vector()[:] = v_std[obs_local][dofmap]

        u_obs_pts.vector().apply("insert")
        v_obs_pts.vector().apply("insert")
        u_std_pts.vector().apply("insert")
        v_std_pts.vector().apply("insert")

        # Interpolate from model
        u_pts = Function(obs_space, name='u_pts')
        v_pts = Function(obs_space, name='v_pts')

        # Project modelled velocity to DG1 to simplify graph coloring
        interp_space = FunctionSpace(self.mesh, 'DG', 1)
        uf = project(u, interp_space)
        vf = project(v, interp_space)

        uf.rename("uf", "")
        vf.rename("vf", "")

        # TODO - what's the significance of missing some points here?
        # Does this affect the value of the cost function?
        interper2 = InterpolationSolver(uf, u_pts, tolerance=1.0)
        interper2.solve()

        P = interper2._B[0]._A._P
        P_T = interper2._B[0]._A._P_T
        InterpolationSolver(vf, v_pts, P=P, P_T=P_T, tolerance=1.0).solve()

        J = Functional(name="J")

        # Continuous
        # data misfit component of J (Isaac 12), with
        # diagonal noise covariance matrix
        # J_ls = lambda_a*(u_std**(-2.0)*(u-u_obs)**2.0 + v_std**(-2.0)*\
        # (v-v_obs)**2.0)*self.dObs

        # Inner product
        J_ls_term_u = new_real_function(name="J_term_u")
        J_ls_term_v = new_real_function(name="J_term_v")

        u_mismatch = ((u_pts-u_obs_pts)/u_std_pts)
        NormSqSolver(project(u_mismatch, obs_space), J_ls_term_u).solve()
        v_mismatch = ((v_pts-v_obs_pts)/v_std_pts)
        NormSqSolver(project(v_mismatch, obs_space), J_ls_term_v).solve()

        # J_ls_term_final = new_real_function()
        # ExprEvaluationSolver(J_ls_term * \
        # lambda_a, J_ls_term_final).solve()

        J.addto(J_ls_term_u)
        J.addto(J_ls_term_v)

        # Regularization
        Prior = self.model.get_prior()
        lap = Prior(self, self.Qp)

        J_reg_alpha, J_reg_beta = lap.J_reg(alpha=alpha, beta=beta, beta_diff=betadiff)

        if do_alpha: J.addto(J_reg_alpha)
        if do_beta: J.addto(J_reg_beta)

        # Get dict of regularisation components for e.g. L-curve analysis
        J_reg_terms = lap.J_reg_terms(alpha=alpha, beta=beta, beta_diff=betadiff)

        # for block in manager()._blocks + [manager()._block]:
        #     for eq in block:
        #         if isinstance(eq, EquationSolver):
        #             solver_parameters = eq._solver_parameters
        #             linear_solver_parameters = eq._linear_solver_parameters
        #             adjoint_solver_parameters = eq._adjoint_solver_parameters
        #             tlm_solver_parameters = eq._tlm_solver_parameters
        #             print(f"Eq: {eq}")
        #             print(f"Solver parameters: {solver_parameters}")
        #             print(f"linear parameters: {linear_solver_parameters}")
        #             print(f"adjoint parameters: {adjoint_solver_parameters}")
        #             print(f"tlm parameters: {tlm_solver_parameters}")

        if verbose:
            J_ls_u = new_real_function(name="J_ls_term_x")
            J_ls_v = new_real_function(name="J_ls_term_y")
            ExprEvaluationSolver(J_ls_term_u, J_ls_u).solve()
            ExprEvaluationSolver(J_ls_term_v, J_ls_v).solve()

            # Print out results
            J1 = J.value()
            J2 = J_ls_u.values()[0] + J_ls_v.values()[0]

            # Assemble & write out terms of regularisation term
            J3 = 0.0
            J_fields = {}
            for term in J_reg_terms:
                assembled = assemble(J_reg_terms[term])
                J3 += assembled
                J_fields[f"J_{term}"] = assembled

            # Add params & full J terms to dict
            J_fields = {**J_fields, **{"delta_alpha": self.delta_alpha,
                                       "gamma_alpha": self.gamma_alpha,
                                       "delta_beta": self.delta_beta,
                                       "delta_beta_gnd": self.delta_beta_gnd,
                                       "gamma_beta": self.gamma_beta,
                                       "J": J1,
                                       "J_ls": J2,
                                       "J_reg": J3}}

            # Additional separate regularisation terms
            for term in J_reg_terms:
                J_fields[f"J_{term}"] = assemble(J_reg_terms[term])

            info('Inversion Details')
            for key in J_fields:
                info(f"{key}: {J_fields[key]}")
            info('')

            inout.dict_to_csv(J_fields, 'Js', self.params)

        return J

    def comp_Q_vaf(self, verbose=False):
        """QOI: Volume above flotation"""
        cnst = self.params.constants

        H = self.H_np
        bed = self.bed
        rhoi = Constant(cnst.rhoi, name="Constant rhoi")
        rhow = Constant(cnst.rhow, name="Constant rhow")
        dIce = self.dIce

        b_ex = conditional(bed < 0.0, 1.0, 0.0)
        HAF = ufl.Max(b_ex * (H + (rhow/rhoi)*bed) + (1-b_ex)*(H), 0.0)

        Q_vaf = HAF * dIce

        if verbose:
            info(f"Q_vaf: {assemble(Q_vaf)}")

        return Q_vaf

    def comp_Q_h2(self, verbose=False):
        """QOI: Square integral of thickness"""
        Q_h2 = self.H_np * self.H_np * self.dIce
        if verbose:
            info(f"Q_h2: {assemble(Q_h2)}")

        return Q_h2

    # Unused?
    def set_dQ_vaf(self, cntrl):
        Q_vaf = self.timestep(adjoint_flag=1, qoi_func=self.comp_Q_vaf)
        dQ = compute_gradient(Q_vaf, cntrl)
        self.dQ_ts = dQ

    # Unused?
    def set_dQ_h2(self, cntrl):
        Q_h2 = self.timestep(adjoint_flag=1, qoi_func=self.comp_Q_h2)
        dQ = compute_gradient(Q_h2, cntrl)
        self.dQ_ts = dQ

    def set_hessian_action(self, cntrl):
        """
        Construct the Hessian object (defined by tlm_adjoint)
        with the functional J
        """
        if type(cntrl) is not list:
            cntrl = [cntrl]
        assert len(cntrl) == 1

        reset_manager()
        clear_caches()
        start_manager()
        J = self.forward(cntrl[0])
        stop_manager()

        self.ddJ = SingleBlockHessian(J)

    def save_ts_zero(self):
        self.H_init = Function(self.H_np.function_space())
        self.U_init = Function(self.U.function_space())

        self.H_init.assign(self.H, annotate=False)
        self.U_init.assign(self.U, annotate=False)

        self.H_init.rename("H_init", "")
        self.U_init.rename("U_init", "")

    def reset_ts_zero(self):
        self.U.assign(self.U_init, annotate=False)
        self.U_np.assign(self.U_init, annotate=False)
        self.H_np.assign(self.H_init, annotate=False)
        self.H.assign(self.H_init, annotate=False)

# TODO - this isn't referenced anywhere
class ddJ_wrapper(object):
    def __init__(self, ddJ_action, cntrl):
        self.ddJ_action = ddJ_action
        self.ddJ_F = Function(cntrl.function_space())

    def apply(self, x):
        with x.getBuffer(readonly=True) as x_a:
            self.ddJ_F.vector().set_local(x_a)
        self.ddJ_F.vector().apply('insert')
        return self.ddJ_action(self.ddJ_F).vector().get_local()


class MomentumSolver(EquationSolver):

    def __init__(self, *args, **kwargs):
        self.picard_params = kwargs.pop("picard_params", None)
        self.J_p = kwargs.pop("J_p", None)
        super(MomentumSolver, self).__init__(*args, **kwargs)

    def drop_references(self):
        super().drop_references()
        self.J_p = replaced_form(self.J_p)

    def forward_solve(self, x, deps=None):
        if deps is None:
            deps = self.dependencies()
            def replace_deps(form): return form  # noqa: E704
        else:
            from collections import OrderedDict
            replace_map = OrderedDict(zip(self.dependencies(), deps))
            replace_map[self.x()] = x
            def replace_deps(form): return ufl.replace(form, replace_map)  # noqa: E704
            # for i, (dep_x, dep) in enumerate(zip(self.dependencies(), deps)):
            # info("%i %s %.16e" % (i, dep_x.name(), dep.vector().norm("l2")))
        if self._initial_guess_index is not None:
            function_assign(x, deps[self._initial_guess_index])

        for i, bc in enumerate(self._bcs):
            keys = bc.get_boundary_values().keys()
            values = bc.get_boundary_values().items()
            keys = list(keys)
            import numpy
            values = numpy.array(list(values))
            info("BC %i %i %.16e" % (i, len(keys), (values * values).sum()))

        lhs = replace_deps(self._lhs)
        rhs = 0 if self._rhs == 0 else replace_deps(self._rhs)
        J_p = replace_deps(self.J_p)
        J = replace_deps(self._J)
        # First order approx - inconsistent jacobian
        # 'replace_deps' is only used by forward replay - tlm_adjoint stuff
        solve(lhs == rhs, x, self._bcs, J=J_p,
              form_compiler_parameters=self._form_compiler_parameters,
              solver_parameters=self.picard_params)
        end()

        # Newton solver
        solve(lhs == rhs, x, self._bcs, J=J,
              form_compiler_parameters=self._form_compiler_parameters,
              solver_parameters=self._solver_parameters)
        end()
