import timeit
import time
from pathlib import Path
import numpy as np

from fenics import *
from tlm_adjoint_fenics import *
from tlm_adjoint_fenics.hessian_optimization import *
#from dolfin_adjoint import *
#from dolfin_adjoint_custom import EquationSolver
import ufl




class ssa_solver:

    def __init__(self, model):
        # Enable aggressive compiler options
        parameters["form_compiler"]["optimize"] = False
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math -march=native"


        self.model = model
        self.params = model.params

        #Fields
        self.bed = model.bed
        self.H_np = model.H_np
        self.H_s = model.H_s
        self.H = model.H
        #self.surf = model.surf
        self.beta = model.beta
        self.beta_bgd = model.beta_bgd
        self.mask = model.mask
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


        #Parameterization of alpha/beta
        self.bglen_to_beta = model.bglen_to_beta
        self.beta_to_bglen = model.beta_to_bglen

        #Facet normals
        self.nm = model.nm

        #Save observations for inversions
        try:
            self.u_obs = model.u_obs
            self.v_obs = model.v_obs
            self.u_std = model.u_std
            self.v_std = model.v_std
            self.uv_obs_pts = model.uv_obs_pts
        except:
            pass

        #Mesh/Function Spaces
        self.mesh = model.mesh
        self.V = model.V
        self.Q = model.Q
        self.Qp = model.Qp
        self.M = model.M
        self.RT = model.RT

        #Trial/Test Functions
        self.U = Function(self.V, name = "U")
        self.U_np = Function(self.V, name = "U_np")
        self.Phi = TestFunction(self.V)
        self.Ksi = TestFunction(self.M)
        self.pTau = TestFunction(self.Qp)

        self.trial_H = TrialFunction(self.M)
        self.H_nps = Function(self.M)

        #Cells
        self.cf = model.cf
        self.OMEGA_DEF = model.OMEGA_DEF
        self.OMEGA_ICE_FLT = model.OMEGA_ICE_FLT
        self.OMEGA_ICE_GND = model.OMEGA_ICE_GND
        self.OMEGA_ICE_FLT_OBS = model.OMEGA_ICE_FLT_OBS
        self.OMEGA_ICE_GND_OBS = model.OMEGA_ICE_GND_OBS

        #Facets
        self.ff = model.ff
        self.GAMMA_DEF = model.GAMMA_DEF
        self.GAMMA_LAT = model.GAMMA_LAT
        self.GAMMA_TMN = model.GAMMA_TMN      #Value at ice terminus
        self.GAMMA_NF = model.GAMMA_NF


        #Measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.cf)
        self.dS = Measure('dS', domain=self.mesh, subdomain_data=self.ff)
        self.ds = ufl.ds

        self.dIce = self.dx
        self.dIce_flt = self.dx(self.OMEGA_ICE_FLT) + self.dx(self.OMEGA_ICE_FLT_OBS)
        self.dIce_gnd = self.dx(self.OMEGA_ICE_GND) + self.dx(self.OMEGA_ICE_GND_OBS)

        self.dObs = self.dx(self.OMEGA_ICE_FLT_OBS) + self.dx(self.OMEGA_ICE_GND_OBS)
        self.dObs_gnd = self.dx(self.OMEGA_ICE_FLT_OBS)
        self.dObs_flt = self.dx(self.OMEGA_ICE_GND_OBS)

        self.dt = Constant(self.params.time.dt)


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
        qoi_dict = {'vaf':self.comp_Q_vaf, 'h2':self.comp_Q_h2}
        choice = self.params.error_prop.qoi
        return qoi_dict[choice.lower()] #flexible case

    def def_mom_eq(self):

        #Simplify accessing fields and parameters
        constants = self.params.constants
        bed = self.bed
        H = self.H
        mask = self.mask
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

        #Vector components of trial function
        u, v = split(self.U)

        #Vector components of test function
        Phi = self.Phi
        Phi_x, Phi_y = split(Phi)

        #Derivatives
        u_x, u_y = u.dx(0), u.dx(1)
        v_x, v_y = v.dx(0), v.dx(1)

        #Viscosity
        U_marker = Function(self.U.function_space(), name = "%s_marker" % self.U.name())
        nu = self.viscosity(U_marker)


        #Switch parameters
        H_s = -rhow/rhoi * bed
        fl_ex = ufl.operators.Conditional(H <= H_s, Constant(1.0), Constant(0.0))

        B2 = self.sliding_law(alpha,U_marker)

        #Driving stress quantities
        F = (1 - fl_ex) * 0.5*rhoi*g*H**2 + \
            (fl_ex) * 0.5*rhoi*g*(delta*H**2 + (1-delta)*H_s**2 )

        W = (1 - fl_ex) * rhoi*g*H + \
            (fl_ex) * rhoi*g*H_s

        draft = (fl_ex) * (rhoi / rhow) * H

        #Terminating margin boundary condition
        sigma_n = 0.5 * rhoi * g * ((H ** 2) - (rhow / rhoi) * (draft ** 2)) - F

        self.mom_F = (
                #Membrance Stresses
                -inner(grad(Phi_x), H * nu * as_vector([4 * u_x + 2 * v_y, u_y + v_x])) * self.dIce
                - inner(grad(Phi_y), H * nu * as_vector([u_y + v_x, 4 * v_y + 2 * u_x])) * self.dIce

                #Basal Drag
                - inner(Phi, (1.0 - fl_ex) * B2 * as_vector([u,v])) * self.dIce

                #Driving Stress
                + ( div(Phi)*F - inner(grad(bed),W*Phi) ) * self.dIce

                #Boundary condition
                + inner(Phi * sigma_n, self.nm) * self.ds )


        # Note - could wrap these 3 statements in ufl.algorithms.expand_derivatives()
        self.mom_Jac_p = ufl.replace(derivative(self.mom_F, self.U), {U_marker:self.U})
        self.mom_F = ufl.replace(self.mom_F, {U_marker:self.U})
        self.mom_Jac = derivative(self.mom_F, self.U)

    def sliding_law(self,alpha,U):

        constants = self.params.constants

        bed = self.bed
        H = self.H
        rhoi = constants.rhoi
        rhow = constants.rhow
        g = constants.g
        sl = self.params.ice_dynamics.sliding_law
        vel_rp = constants.vel_rp

        H_s = -rhow/rhoi * bed
        fl_ex = ufl.operators.Conditional(H <= H_s, Constant(1.0), Constant(0.0))

        C = alpha*alpha
        u,v = split(U)

        if sl == 'linear':
            B2 = C

        elif sl == 'weertman':
            N = (1-fl_ex)*(H*rhoi*g + ufl.Min(bed,0.0)*rhow*g)
            U_mag = sqrt(U[0]**2 + U[1]**2 + vel_rp**2)
            B2 = (1-fl_ex)*(C * N**(1.0/3.0) * U_mag**(-2.0/3.0))

        return B2

    def solve_mom_eq(self, annotate_flag=None):
        #Dirichlet Boundary Conditons: Zero flow

        self.bcs = []

        if not self.params.mesh.periodic_bc:
            ff_array = self.ff.array()
            bc0 = DirichletBC(self.V, self.latbc, self.ff, self.GAMMA_LAT) if self.GAMMA_LAT in ff_array else False
            bc1 = DirichletBC(self.V, (0.0, 0.0), self.ff, self.GAMMA_NF) if self.GAMMA_NF in ff_array else False

            for j in [bc0,bc1]:
                if j: self.bcs.append(j)


        t0 = time.time()

        newton_params = self.params.momsolve.newton_params
        picard_params = self.params.momsolve.picard_params
        J_p = self.mom_Jac_p
        MomentumSolver(self.mom_F == 0, self.U, bcs = self.bcs, J_p=J_p, picard_params = picard_params, solver_parameters = newton_params).solve(annotate=annotate_flag)

        t1 = time.time()
        info("Time for solve: {0}".format(t1-t0))


    def def_thickadv_eq(self):
        U = self.U
        U_np = self.U_np
        Ksi = self.Ksi
        trial_H = self.trial_H
        H_np = self.H_np
        H_s = self.H_s
        H = self.H
        H_init = self.H_init
        bmelt = self.bmelt
        smb = self.smb
        dt = self.dt
        nm = self.nm
        dIce = self.dIce
        dIce_flt = self.dIce_flt
        ds = self.ds
        dS = self.dS

        #Crank Nicholson
        # self.thickadv = (inner(Ksi, ((trial_H - H_np) / dt)) * dIce
        # - inner(grad(Ksi), U_np * 0.5 * (trial_H + H_np)) * dIce
        # + inner(jump(Ksi), jump(0.5 * (dot(U_np, nm) + abs(dot(U_np, nm))) * 0.5 * (trial_H + H_np))) * dS
        # + conditional(dot(U_np, nm) > 0, 1.0, 0.0)*inner(Ksi, dot(U_np * 0.5 * (trial_H + H_np), nm))*ds #Outflow
        # + conditional(dot(U_np, nm) < 0, 1.0 , 0.0)*inner(Ksi, dot(U_np * H_init, nm))*ds #Inflow
        # + bmelt*Ksi*dIce_flt) #basal melting

        #Backward Euler
        self.thickadv = (inner(Ksi, ((trial_H - H_np) / dt)) * dIce
        - inner(grad(Ksi), U_np * trial_H) * dIce
        + inner(jump(Ksi), jump(0.5 * (dot(U_np, nm) + abs(dot(U_np, nm))) * trial_H)) * dS
        + conditional(dot(U_np, nm) > 0, 1.0, 0.0)*inner(Ksi, dot(U_np * trial_H, nm))*ds #Outflow at boundaries
        + conditional(dot(U_np, nm) < 0, 1.0 , 0.0)*inner(Ksi, dot(U_np * H_init, nm))*ds #Inflow at boundaries
        + bmelt*Ksi*dIce_flt #basal melting
        - smb*Ksi*dIce) #surface mass balance

        # #Forward euler
        # self.thickadv = (inner(Ksi, ((trial_H - H_np) / dt)) * dIce
        # - inner(grad(Ksi), U_np * H_np) * dIce
        # + inner(jump(Ksi), jump(0.5 * (dot(U_np, nm) + abs(dot(U_np, nm))) * H_np)) * dS
        # + conditional(dot(U_np, nm) > 0, 1.0, 0.0)*inner(Ksi, dot(U_np * H_np, nm))*ds #Outflow
        # + conditional(dot(U_np, nm) < 0, 1.0 , 0.0)*inner(Ksi, dot(U_np * H_init, nm))*ds #Inflow
        # + bmelt*Ksi*dIce_flt) #basal melting

        self.thickadv_split = ufl.replace(self.thickadv, {U_np:0.5 * (self.U + self.U_np)})

        self.H_bcs = []

    def solve_thickadv_eq(self):

        H_s = self.H_s
        a, L = lhs(self.thickadv), rhs(self.thickadv)
        solve(a==L,H_s,bcs = self.H_bcs)

    def solve_thickadv_split_eq(self):
        H_nps = self.H_nps
        a, L = lhs(self.thickadv_split), rhs(self.thickadv_split)
        solve(a==L,H_nps, bcs = self.H_bcs)

    def timestep(self, save=1, adjoint_flag=1, qoi_func= None ):
        """
        Time evolving model
        Returns the QoI
        """

        #Read timestep info
        config = self.params.time
        n_steps = config.total_steps
        dt = config.dt
        run_length = config.run_length

        outdir = self.params.io.output_dir

        t = 0.0

        self.Qval_ts = np.zeros(n_steps+1)
        Q = Functional()
        Q_is = []

        U = self.U
        U_np = self.U_np
        H = self.H
        H_np = self.H_np
        H_s = self.H_s
        H_nps = self.H_nps


        if adjoint_flag:
            num_sens = self.params.time.num_sens
            t_sens = np.array([run_length]) if num_sens == 1 else np.linspace(0.0, run_length, num_sens)
            n_sens = np.round(t_sens/dt)

            reset()
            start_annotating()
#            configure_checkpointing("periodic_disk", {'period': 2, "format":"pickle"})
            configure_checkpointing("multistage", {"blocks":n_steps,
                                                   "snaps_on_disk":4000,
                                                   "snaps_in_ram":10,
                                                   "verbose":True,
                                                   "format":"pickle"})

        self.def_thickadv_eq()
        self.def_mom_eq()
        self.solve_mom_eq()
        U_np.assign(U)

        if qoi_func is not None:
            qoi = qoi_func()
            self.Qval_ts[0] = assemble(qoi)

        if adjoint_flag:
            if 0.0 in n_sens:
                Q_i = Functional()
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



        #Main timestepping loop
        #=======================
        for n in range(n_steps):
            begin("Starting timestep %i of %i, time = %.16e a" % (n + 1, n_steps, t))

            # Solve

            #Operator splitting
            self.solve_thickadv_eq()
            self.solve_mom_eq()
            self.solve_thickadv_split_eq()

            U_np.assign(U)
            H_np.assign(H_nps)

            # Simple Scheme
            # self.solve_thickadv_eq()
            # H_np.assign(H_s)
            #
            # self.solve_mom_eq()
            # U_np.assign(U)



            #Increment time
            n += 1
            t = n * float(dt)

            if n < n_steps - 1 and adjoint_flag:
                new_block()

                #Record
            if qoi_func is not None:
                qoi = qoi_func()
                self.Qval_ts[n] = assemble(qoi)

                if adjoint_flag:
                    if n in n_sens:
                        Q_i = Functional()
                        Q_i.assign(qoi)
                        Q_is.append(Q_i)
                        Q.addto(Q_i.fn())
                    else:
                        Q.addto()

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

        #If we're using a lumped mass approach, f is alpha_l, not alpha
        if self.lumpedmass_inversion:
            self.alpha = Function(f.function_space(), name='alpha')
            LumpedMassSolver(f, self.alpha, p=-0.5).solve()
            #TODO - 'boundary_correct(self.alpha)'?
        else:
            self.alpha = f

        # if not self.test_outfile:
        #     self.test_outfile = File(os.path.join('invoutput_data','alpha_test.pvd'))
        # self.test_outfile << self.alpha

        self.def_mom_eq()
        self.solve_mom_eq()
        J = self.comp_J_inv()
        return J

    def forward_beta(self, f):
        """
        Runs the forward model w/ given beta (f)
        and returns the cost function J
        """
        clear_caches()
        self.beta = f
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
            info('Inversion iteration: {0}/{1}'.format(j+1,num_iter) )

            cntrl = cntrl_input[j % nparam]
            if cntrl.name() == 'alpha':
                cc = self.alpha
                if self.lumpedmass_inversion:
                    cc = self.alpha_l

                forward = self.forward_alpha

            else:
                cc = self.beta
                forward = self.forward_beta

            reset()
            clear_caches()
            start_annotating()
            J = forward(cc)
            stop_annotating()

            #dJ = compute_gradient(J, self.alpha)
            #ddJ = Hessian(forward)
            #min_order = taylor_test(forward, self.alpha, J_val = J.value(), dJ = dJ, ddJ=ddJ, seed = 1.0e-6)

            cntrl_opt, result = minimize_scipy(forward, cc, J, method='L-BFGS-B',
                                               options=config.inv_options)
              #options = {"ftol":0.0, "gtol":1.0e-12, "disp":True, 'maxiter': 10})

            cc.assign(cntrl_opt)

        self.def_mom_eq()

        #Re-compute velocities with inversion results
        reset()
        clear_caches()
        start_annotating()
        self.solve_mom_eq()
        stop_annotating()

        #Print out inversion results/parameter values
        self.J_inv = self.comp_J_inv(verbose=True)


    def epsilon(self, U):
        """
        return the strain-rate tensor of self.U.
        """
        epsdot = sym(grad(U))
        return epsdot

    def effective_strain_rate(self, U):
        """
        return the effective strain rate squared.
        """
        eps_rp = self.params.constants.eps_rp

        eps = self.epsilon(U)
        exx = eps[0,0]
        eyy = eps[1,1]
        exy = eps[0,1]

        # Second invariant of the strain rate tensor squared
        eps_2 = (exx**2 + eyy**2 + exx*eyy + (exy)**2 + eps_rp**2)

        return eps_2

    def viscosity(self,U):
        B = self.beta_to_bglen(self.beta)
        n = self.params.constants.glen_n

        eps_2 = self.effective_strain_rate(U)
        nu = 0.5 * B * eps_2**((1.0-n)/(2.0*n))

        return nu

    def comp_J_inv(self, verbose=False):
        """
        Compute the value of the cost function
        Note: 'verbose' significantly decreases speed
        """

        invconfig = self.params.inversion

        #What are we inverting for?:
        do_alpha = invconfig.alpha_active
        do_beta = invconfig.beta_active

        u,v = split(self.U)

        # Observed velocities
        u_obs = self.u_obs
        v_obs = self.v_obs
        u_std = self.u_std
        v_std = self.v_std
        uv_obs_pts = self.uv_obs_pts

        # Determine observations within our mesh partition
        # TODO find a faster way to do this
        cell_max = self.mesh.cells().shape[0]
        obs_local = np.zeros_like(u_obs, dtype=np.bool)
        for i, pt in enumerate(uv_obs_pts):
            obs_local[i] = self.mesh.bounding_box_tree().\
                compute_first_entity_collision(Point(pt)) <= cell_max

        local_cnt = np.sum(obs_local)

        alpha = self.alpha
        beta = self.beta
        beta_bgd = self.beta_bgd
        betadiff = beta-beta_bgd

        dIce = self.dIce
        dIce_gnd = self.dIce_gnd
        ds = self.ds
        nm = self.nm

        #regularization parameters
        delta_a = self.delta_alpha
        delta_b = self.delta_beta
        gamma_a = self.gamma_alpha
        gamma_b = self.gamma_beta

        # Sample Discrete Points
        
        #Arbitrary mesh to define function for interpolated variables
        obs_mesh = UnitIntervalMesh(MPI.comm_self, local_cnt)
        obs_space = FunctionSpace(obs_mesh, "Discontinuous Lagrange", 0)
        u_obs_pts = Function(obs_space, name='u_obs_pts')
        v_obs_pts = Function(obs_space, name='v_obs_pts')
        u_std_pts = Function(obs_space, name='u_std_pts')
        v_std_pts = Function(obs_space, name='v_std_pts')

        u_obs_pts.vector()[:] = u_obs[obs_local]
        v_obs_pts.vector()[:] = v_obs[obs_local]
        u_std_pts.vector()[:] = u_std[obs_local]
        v_std_pts.vector()[:] = v_std[obs_local]

        u_obs_pts.vector().apply("insert")
        v_obs_pts.vector().apply("insert")
        u_std_pts.vector().apply("insert")
        v_std_pts.vector().apply("insert")

        # Interpolate from model
        u_pts = Function(obs_space, name='u_pts')
        v_pts = Function(obs_space, name='v_pts')

        # TODO - is projection to M instead of Q OK here?
        # it's necessary for InterpolationSolver to work
        uf = project(u, self.M)
        vf = project(v, self.M)

        uf.rename("uf", "")
        vf.rename("vf", "")

        interper2 = InterpolationSolver(uf,
                                        u_pts,
                                        X_coords=uv_obs_pts[obs_local])
        interper2.solve()
        P = interper2._B[0]._A._P
        P_T = interper2._B[0]._A._P_T
        InterpolationSolver(vf, v_pts, P=P, P_T=P_T).solve()

        J = Functional(name="J")

        # Continuous
        # data misfit component of J (Isaac 12), with
        # diagonal noise covariance matrix
        # J_ls = lambda_a*(u_std**(-2.0)*(u-u_obs)**2.0 + v_std**(-2.0)*\
        # (v-v_obs)**2.0)*self.dObs

        # Inner product
        J_ls_term_new = new_real_function(name=f"J_term")

        # Result of NormSqSolver is added to J_ls_term_new
        # so calling twice is a sum
        u_mismatch = ((u_pts-u_obs_pts)/u_std_pts)
        NormSqSolver(project(u_mismatch, obs_space), J_ls_term_new).solve()
        v_mismatch = ((v_pts-v_obs_pts)/v_std_pts)
        NormSqSolver(project(v_mismatch, obs_space), J_ls_term_new).solve()

        # J_ls_term_final = new_real_function()
        # ExprEvaluationSolver(J_ls_term_new * \
        # lambda_a, J_ls_term_final).solve()

        J.addto(J_ls_term_new)

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
            L = (delta_a * alpha * self.pTau -
                 gamma_a*inner(grad(alpha), grad(self.pTau)))*dIce
            solve(a == L, f_alpha)
            J_reg_alpha = inner(f_alpha, f_alpha)*dIce
            J.addto(J_reg_alpha)

            # if not self.f_alpha_file:
            #     self.f_alpha_file = \
            # File(os.path.join('invoutput_data', 'f_alpha_test.pvd'))
            # self.f_alpha_file << f_alpha

        if(do_beta):
            L = (delta_b * betadiff * self.pTau - \
                 gamma_b*inner(grad(betadiff), grad(self.pTau)))*dIce
            solve(a == L, f_beta)
            J_reg_beta = inner(f_beta, f_beta)*dIce
            J.addto(J_reg_beta)

        # Continuous
        # J = J_ls + J_reg_alpha + J_reg_beta

        if verbose:
            J_ls = new_real_function(name="J_ls_term")
            ExprEvaluationSolver(J_ls_term_new, J_ls).solve()

            # Print out results
            J1 = J.value()
            J2 = J_ls.values()[0]
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

        print("WARNING: Think comp_Q_vaf returns zero on first timestep - ",
              "check initialisation of H_nps")
        cnst = self.params.constants

        H = self.H_nps
        # B stands in for self.bed, which leads to a taping error
        B = Function(self.M)
        B.assign(self.bed, annotate=False)
        rhoi = Constant(cnst.rhoi)
        rhow = Constant(cnst.rhow)
        dIce = self.dIce
        dIce_gnd = self.dIce_gnd
        dt = self.dt

        b_ex = conditional(B < 0.0, 1.0, 0.0)
        HAF = b_ex * (H + rhow/rhoi*B) + (1-b_ex)*(H)
        Q_vaf = HAF * dIce_gnd

        if verbose: print('Q_vaf: {0}'.format(Q_vaf))

        return Q_vaf

    def comp_Q_h2(self, verbose=False):
        """
        QOI: Square integral of thickness
        """

        Q_h2 = self.H_np * self.H_np * self.dIce
        if verbose: print('Q_h2: {0}'.format(Q_h2))

        return Q_h2

    def set_dQ_vaf(self, cntrl):
        Q_vaf = self.timestep(adjoint_flag=1, qoi_func=self.comp_Q_vaf)
        dQ = compute_gradient(Q_vaf, cntrl)
        self.dQ_ts = dQ

    def set_dQ_h2(self, cntrl):
        Q_h2 = self.timestep(adjoint_flag=1, qoi_func=self.comp_Q_h2)
        dQ = compute_gradient(Q_h2, cntrl)
        self.dQ_ts = dQ

    def set_hessian_action(self, cntrl):
        """
        Construct the Hessian object (defined by tlm_adjoint)
        with the functional J
        """
        if type(cntrl) is not list: cntrl = [cntrl]
        fopts = {'alpha': self.forward_alpha,
                 'beta': self.forward_beta,
                 'dual': self.forward_dual}

        forward = fopts['dual'] if len(cntrl) > 1 else fopts[cntrl[0].name()]

        reset()
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

    def reset_ts_zero(self):
        self.U.assign(self.U_init, annotate=False)
        self.U_np.assign(self.U_init, annotate=False)
        self.H_np.assign(self.H_init, annotate=False)
        self.H_s.assign(self.H_init, annotate=False)
        self.H_nps.assign(self.H_init, annotate=False)
        self.H = 0.5*(self.H_np + self.H_s)

#TODO - this isn't referenced anywhere
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

    def replace(self, replace_map):
        super(MomentumSolver, self).replace(replace_map)
        self.J_p = ufl.replace(self.J_p, replace_map)

    def forward_solve(self, x, deps=None):
        if deps is None:
            deps = self.dependencies()
            replace_deps = lambda form: form
        else:
            from collections import OrderedDict
            replace_map = OrderedDict(zip(self.dependencies(), deps))
            replace_map[self.x()] = x
            replace_deps = lambda form: ufl.replace(form, replace_map)
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
        solve(lhs == rhs, x, self._bcs, J=J_p,
              form_compiler_parameters=self._form_compiler_parameters,
              solver_parameters=self.picard_params)
        end()
        solve(lhs == rhs, x, self._bcs, J=J,
              form_compiler_parameters=self._form_compiler_parameters,
              solver_parameters=self._solver_parameters)
        end()


#=======================================================
# Lumped Mass Matrix stuff for variable mesh resolution
#=======================================================
def mass_matrix_diagonal(space, name="M_l"):
    M_l = Function(space, name=name)
    M_l.vector().axpy(1.0, assemble(TestFunction(space) * dx))
    return M_l


class LumpedMassSolver(Equation):
    def __init__(self, m, m_l, p=1):
        Equation.__init__(self, m_l, deps=[m_l, m], nl_deps=[], ic_deps=[],
                          adj_ic_deps=[])
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
