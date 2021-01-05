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
from fenics_ice import inout
from tlm_adjoint_fenics import *
from tlm_adjoint_fenics.hessian_optimization import *
# from dolfin_adjoint import *
# from dolfin_adjoint_custom import EquationSolver
import ufl
import logging

log = logging.getLogger("fenics_ice")

class ssa_solver:

    def __init__(self, model):
        # Enable aggressive compiler options
        parameters["form_compiler"]["optimize"] = False
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math -march=native"
        parameters["form_compiler"]["precision"] = 16

        self.model = model
        self.model.solvers.append(self)
        self.params = model.params

        # Fields
        self.bed = model.bed
        self.H_np = model.H_np
        self.H = model.H
        self.beta = model.beta
        self.beta_bgd = model.beta_bgd
        self.alpha = model.alpha
        self.bmelt = model.bmelt
        self.smb = model.smb
        self.latbc = model.latbc

        self.set_inv_params()

        # self.test_outfile = None
        # self.f_alpha_file = None

        self.lumpedmass_inversion = False
        if self.lumpedmass_inversion:
            self.alpha_l = Function(self.alpha.function_space())
            LumpedMassSolver(self.alpha, self.alpha_l, p=0.5).solve(annotate=False, tlm=False)


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

        # Mesh/Function Spaces
        self.mesh = model.mesh
        self.V = model.V
        self.Q = model.Q
        self.Qp = model.Qp
        self.M = model.M
        self.RT = model.RT

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
        self.gamma_beta = invparam.gamma_beta

    def zero_inv_params(self):

        self.delta_alpha = 1E-10
        self.gamma_alpha = 1E-10
        self.delta_beta = 1E-10
        self.gamma_beta = 1E-10

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

        elif sl == 'weertman':
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
                                 "absolute_tolerance": "1e-10",
                                 "relative_tolerance": "1e-11",
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

        manager_info()
        return Q_is if qoi_func is not None else None

    # def forward_ts_alpha(self,aa):
    #     clear_caches()
    #     self.timestep()
    #     new_block()
    #     self.Q_vaf = self.comp_Q_vaf()
    #     Q_vaf = Functional()
    #     Q_vaf.assign(self.Q_vaf)
    #     return Q_vaf

    def forward_alpha(self, f):
        """
        Runs the forward model w/ given alpha (f)
        and returns the cost function J
        """
        clear_caches()

        # If we're using a lumped mass approach, f is alpha_l, not alpha
        if self.lumpedmass_inversion:
            self.alpha = Function(f.function_space(), name='alpha')
            LumpedMassSolver(f, self.alpha, p=-0.5).solve()
            # TODO - 'boundary_correct(self.alpha)'?
        else:
            self.alpha = f
            self.alpha.rename("alpha", "")

        # if not self.test_outfile:
        #     self.test_outfile = File(os.path.join('invoutput_data','alpha_test.pvd'))
        # self.test_outfile << self.alpha

        self.def_mom_eq()
        self.solve_mom_eq()
        J = self.comp_J_inv()  # TODO - make verbose TOML configurable
        return J

    def forward_beta(self, f):
        """
        Runs the forward model w/ given beta (f)
        and returns the cost function J
        """
        clear_caches()
        self.beta = f
        self.beta.rename("beta", "")
        self.def_mom_eq()
        self.solve_mom_eq()
        J = self.comp_J_inv(verbose=True)
        return J

    def forward_dual(self, f):
        """
        Runs the forward model w/ given
        alpha and beta (f[0], f[1])
        and returns the cost function J
        """
        clear_caches()
        self.alpha = f[0]
        self.beta = f[1]

        self.alpha.rename("alpha","")
        self.beta.rename("beta","")

        self.def_mom_eq()
        self.solve_mom_eq()
        J = self.comp_J_inv(verbose=True)
        return J

    def get_control(self):
        """
        Returns the list (length 1 or 2) of
        control params (i.e. alpha and/or beta)
        """
        config = self.params.inversion
        cntrl = []
        if config.alpha_active:
            cntrl.append(self.alpha)
        if config.beta_active:
            cntrl.append(self.beta)

        return cntrl

    def inversion(self):

        config = self.params.inversion

        cntrl_input = self.get_control()
        nparam = len(cntrl_input)

        num_iter = config.alt_iter*nparam if nparam > 1 else nparam

        for j in range(num_iter):
            info('Inversion iteration: {0}/{1}'.format(j+1, num_iter) )

            cntrl = cntrl_input[j % nparam]
            if cntrl.name() == 'alpha':
                cc = self.alpha
                if self.lumpedmass_inversion:
                    cc = self.alpha_l

                forward = self.forward_alpha

            else:
                cc = self.beta
                forward = self.forward_beta

            reset_manager()
            clear_caches()
            start_annotating()
            J = forward(cc)
            stop_annotating()

            # dJ = compute_gradient(J, self.alpha)
            # ddJ = Hessian(forward)
            # min_order = taylor_test(forward, self.alpha, J_val=J.value(),
            #                         dJ=dJ, ddJ=ddJ, seed=1.0e-6)

            cntrl_opt, result = minimize_scipy(forward, cc, J, method='L-BFGS-B',
                                               options=config.inv_options)
            # options = {"ftol":0.0, "gtol":1.0e-12, "disp":True, 'maxiter': 10})

            cc.assign(cntrl_opt)

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

        # Regularization parameters
        delta_a = self.delta_alpha
        delta_b = self.delta_beta
        gamma_a = self.gamma_alpha
        gamma_b = self.gamma_beta

        # Determine observations within our mesh partition
        # Note that although cell_max counts ghost_cells,
        # compute_first_entity_collision seems to ignore
        # ghost elements, and so arrives at the right answer
        cell_max = self.mesh.cells().shape[0]
        obs_local = np.zeros_like(u_obs, dtype=np.bool)

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

        assert global_vcnt == len(obs_local), ("Mismatch between total number of observation "
        "points and the sum of those assigned to each partition. "
        "Note that this might not necessarily be an error - should be handled better.")

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

        f = TrialFunction(self.Qp)
        f_alpha = Function(self.Qp, name="f_alpha")
        f_beta = Function(self.Qp, name="f_beta")

        # cf. Isaac 5, delta component -> invertiblity, gamma -> smoothness
        a = f*self.pTau*dIce

        if(do_alpha):
            # This L is equivalent to scriptF in reg_operator.pdf
            # Prior.py contains vector equivalent of this
            # (this operates on fem functions)
            L = (delta_a * alpha * self.pTau
                 + gamma_a*inner(grad(alpha), grad(self.pTau)))*dIce
            solve(a == L, f_alpha)
            J_reg_alpha = 0.5 * inner(f_alpha, f_alpha)*dIce
            J.addto(J_reg_alpha)

            # if not self.f_alpha_file:
            #     self.f_alpha_file = \
            # File(os.path.join('invoutput_data', 'f_alpha_test.pvd'))
            # self.f_alpha_file << f_alpha

        if(do_beta):
            L = (delta_b * betadiff * self.pTau
                 + gamma_b*inner(grad(betadiff), grad(self.pTau)))*dIce
            solve(a == L, f_beta)
            J_reg_beta = 0.5 * inner(f_beta, f_beta)*dIce
            J.addto(J_reg_beta)

        # Continuous
        # J = J_ls + J_reg_alpha + J_reg_beta

        if verbose:
            J_ls_u = new_real_function(name="J_ls_term_x")
            J_ls_v = new_real_function(name="J_ls_term_y")
            ExprEvaluationSolver(J_ls_term_u, J_ls_u).solve()
            ExprEvaluationSolver(J_ls_term_v, J_ls_v).solve()

            # Print out results
            J1 = J.value()
            J2 = J_ls_u.values()[0] + J_ls_v.values()[0]
            J3 = assemble(J_reg_alpha) if do_alpha else 0.0
            J4 = assemble(J_reg_beta) if do_beta else 0.0

            info('Inversion Details')
            info('delta_a: %.5e' % delta_a)
            info('delta_b: %.5e' % delta_b)
            info('gamma_a: %.5e' % gamma_a)
            info('gamma_b: %.5e' % gamma_b)
            info('J: %.5e' % J1)
            info('J_ls: %.5e' % J2)
            info('J_reg: %.5e' % sum([J3, J4]))
            info('J_reg_alpha: %.5e' % J3)
            info('J_reg_beta: %.5e' % J4)
            info('J_reg/J_cst: %.5e' % ((J3+J4)/(J2)))
            info('')

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
        fopts = {'alpha': self.forward_alpha,
                 'beta': self.forward_beta,
                 'dual': self.forward_dual}

        forward = fopts['dual'] if len(cntrl) > 1 else fopts[cntrl[0].name()]

        reset_manager()
        clear_caches()
        start_manager()
        J = forward(cntrl[0])
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
        self.ddJ_F.vector().set_local(x.getArray())
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


########################################################
# Lumped Mass Matrix stuff for variable mesh resolution
########################################################
def mass_matrix_diagonal(space, name="M_l"):
    M_l = Function(space, name=name)
    M_l.vector().axpy(1.0, assemble(TestFunction(space) * dx))
    return M_l


class LumpedMassSolver(Equation):
    def __init__(self, m, m_l, p=1):
        Equation.__init__(self, m_l, deps=[m_l, m], nl_deps=[],
                          ic=False, adj_ic=False)
        self._M_l = mass_matrix_diagonal(function_space(m_l))
        self._M_l_p = p

    def forward_solve(self, x, deps=None):
        _, m = self.dependencies() if deps is None else deps
        function_set_values(
            x,
            function_get_values(m)
            * np.power(function_get_values(self._M_l), self._M_l_p))

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            return adj_x
        else:
            assert dep_index == 1
            F = function_new(self._M_l)
            function_set_values(
                F,
                -function_get_values(adj_x)
                * np.power(function_get_values(self._M_l), self._M_l_p))
            return F

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        m_l, m = self.dependencies()
        assert m_l not in M
        return LumpedMassSolver(get_tangent_linear(m, M, dM, tlm_map),
                                tlm_map[m_l], p=self._M_l_p)
