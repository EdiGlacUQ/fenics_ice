import sys
sys.path.insert(0,'../../dolfin_adjoint_custom/python/')
from fenics import *
from tlm_adjoint import *
from tlm_adjoint.hessian_optimization import *
#from dolfin_adjoint import *
#from dolfin_adjoint_custom import EquationSolver
import moola
import numpy as np
import ufl
import os
import timeit
import time
from IPython import embed




class ssa_solver:

    def __init__(self, model):
        # Enable aggressive compiler options
        parameters["form_compiler"]["optimize"] = False
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math -march=native"


        self.model = model
        self.param = model.param

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
        self.latbc = model.latbc

        #Parameterization of alpha/beta
        self.apply_prmz = model.apply_prmz
        self.rev_prmz = model.rev_prmz

        #Facet normals
        self.nm = model.nm

        #Save observations for inversions
        try:
            self.u_obs = model.u_obs
            self.v_obs = model.v_obs
            self.u_std = model.u_std
            self.v_std = model.v_std
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
        self.ds = dolfin.ds

        self.dIce = self.dx
        self.dIce_flt = self.dx(self.OMEGA_ICE_FLT) + self.dx(self.OMEGA_ICE_FLT_OBS)
        self.dIce_gnd = self.dx(self.OMEGA_ICE_GND) + self.dx(self.OMEGA_ICE_GND_OBS)

        self.dObs = self.dx(self.OMEGA_ICE_FLT_OBS) + self.dx(self.OMEGA_ICE_GND_OBS)
        self.dObs_gnd = self.dx(self.OMEGA_ICE_FLT_OBS)
        self.dObs_flt = self.dx(self.OMEGA_ICE_GND_OBS)

        self.dt = Constant(self.param['dt'])



    def def_mom_eq(self):

        #Simplify accessing fields and parameters
        bed = self.bed
        H = self.H
        #surf = self.surf
        mask = self.mask
        alpha = self.alpha


        rhoi = self.param['rhoi']
        rhow = self.param['rhow']
        delta = 1.0 - rhoi/rhow
        g = self.param['g']
        n = self.param['n']
        tol = self.param['tol']
        dIce = self.dIce
        ds = self.ds

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

        #Sliding law
        #B2 = exp(alpha)
        B2 = self.rev_prmz(alpha)

        #Switch parameters
        H_s = -rhow/rhoi * bed
        fl_ex = ufl.operators.Conditional(H <= H_s, Constant(1.0), Constant(0.0))

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


        self.mom_Jac_p = replace(derivative(self.mom_F, self.U), {U_marker:self.U})
        self.mom_F = replace(self.mom_F, {U_marker:self.U})
        self.mom_Jac = derivative(self.mom_F, self.U)

    def solve_mom_eq(self, annotate_flag=None):
        #Dirichlet Boundary Conditons: Zero flow

        self.bcs = []

        if not self.param['periodic_bc']:
            ff_array = self.ff.array()
            bc0 = DirichletBC(self.V, self.latbc, self.ff, self.GAMMA_LAT) if self.GAMMA_LAT in ff_array else False
            bc1 = DirichletBC(self.V, (0.0, 0.0), self.ff, self.GAMMA_NF) if self.GAMMA_NF in ff_array else False

            for j in [bc0,bc1]:
                if j: self.bcs.append(j)


        t0 = time.time()

        newton_params = self.param['newton_params']
        picard_params = self.param['picard_params']
        J_p = self.mom_Jac_p
        MomentumSolver(self.mom_F == 0, self.U, bcs = self.bcs, J_p=J_p, picard_params = picard_params, solver_parameters = newton_params).solve(annotate=annotate_flag)

        t1 = time.time()
        print("Time for solve: ", t1-t0)


    def def_thickadv_eq(self):
        U = self.U
        U_np = self.U_np
        Ksi = self.Ksi
        trial_H = self.trial_H
        H_np = self.H_np
        H_s = self.H_s
        H = self.H
        H_init = self.H_init
        dt = self.dt
        nm = self.nm
        dIce = self.dIce
        ds = self.ds
        dS = self.dS

        self.thickadv = (inner(Ksi, ((trial_H - H_np) / dt)) * dIce
        - inner(grad(Ksi), U_np * 0.5 * (trial_H + H_np)) * dIce
        + inner(jump(Ksi), jump(0.5 * (dot(U_np, nm) + abs(dot(U_np, nm))) * 0.5 * (trial_H + H_np))) * dS
        + inner(Ksi, dot(U_np * 0.5 * (trial_H + H_np), nm)) * ds)
        + conditional(dot(U_np, nm) > 0, inner(Ksi, dot(U_np * 0.5 * (trial_H + H_np), nm)), 0.0)*ds #Outflow
        + conditional(dot(U_np, nm) < 0, inner(Ksi, dot(U_np * 0.5 * H_init, nm)), 0.0)*ds   #Inflow


        self.thickadv_split = replace(self.thickadv, {U_np:0.5 * (self.U + self.U_np)})

        #bc0 = DirichletBC(self.M, self.H_init, self.ff, self.GAMMA_LAT)
        #bc1 = DirichletBC(self.M, (0.0), self.ff, self.GAMMA_TMN)
        #self.H_bcs = [bc0, bc1]
        self.H_bcs = []

    def solve_thickadv_eq(self):

        H_s = self.H_s
        a, L = lhs(self.thickadv), rhs(self.thickadv)
        solve(a==L,H_s,bcs = self.H_bcs)

    def solve_thickadv_split_eq(self):
        H_nps = self.H_nps
        a, L = lhs(self.thickadv_split), rhs(self.thickadv_split)
        solve(a==L,H_nps, bcs = self.H_bcs)

    def timestep(self, save = 1, adjoint_flag=1, qoi_func= None ):


        t = 0.0

        n_steps = self.param['n_steps']
        dt = self.param['dt']
        run_length = self.param['run_length']
        num_sens = self.param['num_sens']


        t_sens = np.array([run_length]) if num_sens == 1 else np.linspace(0.0, run_length, num_sens)
        n_sens = np.round(t_sens/dt)

        outdir = self.param['outdir']

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
            reset()
            start_annotating()
#            configure_checkpointing("periodic_disk", {'period': 2, "format":"pickle"})
            configure_checkpointing("revolve", {"blocks":n_steps, "snaps_on_disk":400, "snaps_in_ram":5, "verbose":True, "format":"pickle"})

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
                Q_i.assign(cst)
                Q_is.append(Q_i)
                Q.addto(Q_i.fn())

            new_block()


        if save:
            hdf_hts = HDF5File(self.mesh.mpi_comm(), os.path.join(outdir, 'H_ts.h5'), 'w')
            hdf_uts = HDF5File(self.mesh.mpi_comm(), os.path.join(outdir, 'U_ts.h5'), 'w')

            pvd_hts = File(os.path.join(outdir, "H_ts.pvd"), "compressed")
            pvd_uts = File(os.path.join(outdir, "U_ts.pvd"), "compressed")

            hdf_hts.write(H_np, 'H', 0.0)
            hdf_uts.write(U_np, 'U', 0.0)

            pvd_hts << (H_np, 0.0)
            pvd_uts << (U_np, 0.0)



        for n in range(n_steps):
            begin("Starting timestep %i of %i, time = %.16e a" % (n + 1, n_steps, t))

            # Solve
            self.solve_thickadv_eq()
            self.solve_mom_eq()
            self.solve_thickadv_split_eq()

            U_np.assign(U)
            H_np.assign(H_nps)

            #Increment time
            n += 1
            t = n * float(dt)

            if n < n_steps - 1 and adjoint_flag:
                new_block()

            #Record
        if qoi_func is not None:
            qoi = qoi_func()
            self.Qval_ts[0] = assemble(qoi)

            if adjoint_flag:
                if n in n_sens:
                    Q_i = Functional()
                    Q_i.assign(cst)
                    Q_is.append(Q_i)
                    Q.addto(Q_i.fn())
                else:
                    Q.addto()

            if save:
                hdf_hts.write(H_np, 'H', t)
                hdf_uts.write(U_np, 'U', t)

                pvd_hts << (H_np, t)
                pvd_uts << (U_np, t)

        return Q_is[0] if qoi_func is not None else None


    def forward_ts_alpha(self,aa):
        clear_caches()
        self.timestep()
        new_block()
        self.J_vaf = self.comp_J_vaf()
        J = Functional()
        J.assign(self.J_vaf)
        return J

    def forward_alpha(self, f):
        clear_caches()
        self.alpha = f
        self.def_mom_eq()
        self.solve_mom_eq()
        self.J_inv = self.comp_J_inv()
        J = Functional()
        J.assign(self.J_inv)
        return J

    def forward_beta(self, f):
        clear_caches()
        self.beta = f
        self.def_mom_eq()
        self.solve_mom_eq()
        self.J_inv = self.comp_J_inv()
        J = Functional()
        J.assign(self.J_inv)
        return J

    def forward_dual(self, dd):
        clear_caches()
        self.alpha = dd[0]
        self.beta = dd[1]
        self.def_mom_eq()
        self.solve_mom_eq()
        self.J_inv = self.comp_J_inv()
        J = Functional()
        J.assign(self.J_inv)
        return J


    def inversion(self, cntrl_input):


        nparam = len(cntrl_input)
        num_iter = self.param['altiter']*nparam if nparam > 1 else nparam

        for j in range(num_iter):

            cntrl = cntrl_input[j % nparam]
            forward = self.forward_alpha if cntrl.name() == 'alpha' else self.forward_beta

            reset()
            clear_caches()
            start_annotating()
            J = forward(cntrl)
            #J = self.forward_alpha(self.alpha)
            stop_annotating()

            #dJ = compute_gradient(J, self.alpha)
            #ddJ = Hessian(forward)
            #min_order = taylor_test(forward, self.alpha, J_val = J.value(), dJ = dJ, ddJ=ddJ, seed = 1.0e-6)

            cntrl_opt, result = minimize_scipy(forward, cntrl, J, method = 'L-BFGS-B',
                options = self.param['inv_options'])
              #options = {"ftol":0.0, "gtol":1.0e-12, "disp":True, 'maxiter': 10})

            cntrl = cntrl_opt

        self.def_mom_eq()


        #Re-compute velocities with inversion results
        reset()
        clear_caches()
        start_annotating()
        self.solve_mom_eq()
        stop_annotating()

        #Print out inversion results/parameter values
        self.J_inv = self.comp_J_inv(verbose=True)


        #
        #
        # if type(cntrl_input) is not list:
        #     cc = Control(cntrl_input)
        #     rf = ReducedFunctional(J, cc, derivative_cb_post = derivative_cb)
        #
        #     ccm = moola.DolfinPrimalVector(cntrl_input)
        #     problem = MoolaOptimizationProblem(rf)
        #     solver = moola.BFGS(problem, ccm, options={'maxiter': self.param['inv_options']['maxiter']})
        #
        #     sol = solver.solve()
        #     opt_var = sol['control']
        #     cntrl_input.vector().set_local(opt_var.array())
        #
        #
        # elif self.param['sim_flag']:
        #
        #     #Control parameters and functional problem
        #     cc = [Control(x) for x in cntrl_input]
        #     rf = ReducedFunctional(J, cc, derivative_cb_post = derivative_cb)
        #
        #     ccm = moola.DolfinPrimalVectorSet([moola.DolfinPrimalVector(x) for x in cntrl_input])
        #
        #     p_scale = []
        #     for i, c in enumerate(cntrl_input):
        #         t0 = time.time()
        #         self.set_hessian_action(c)
        #         A = eigenfunc.HessWrapper(self.ddJ,c)
        #         [lam,v] = eigenfunc.eigens(A,k=10,n_iter=2)
        #         p_scale.append(lam[0])
        #         t1 = time.time()
        #         print("{0}s to determine lead eigenvalue of paramater {1}".format(t1-t0, i))
        #         print('Value: {0}'.format(lam[0]))
        #         adj_reset() #Reset adjoint tape. Emprically necessary
        #         self.def_mom_eq()
        #         self.solve_mom_eq()
        #         self.J_inv = self.comp_J_inv()
        #
        #     Hinit = Hinit_gen(p_scale)
        #
        #     problem = MoolaOptimizationProblem(rf)
        #     solver = moola.BFGS(problem, ccm, options={'jtol': 1e-4,
        #                                            'gtol': 1e-9,
        #                                            'line_search_options' : {"ftol": 1e-4, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1},
        #                                            'Hinit': Hinit,
        #                                            'maxiter': self.param['inv_options']['maxiter'],
        #                                            'mem_lim': 10})
        #
        #     sol = solver.solve()
        #     opt_var = sol['control']
        #
        #
        #     list(map(lambda x: x[0].vector().set_local(x[1].array()), list(zip(cntrl,opt_var))))
        #
        # else:
        #     altiter = self.param['altiter']
        #     nparam = len(cntrl_input)
        #
        #     for j in range(altiter*nparam):
        #         cntrl = cntrl_input[j % nparam]
        #
        #         #Control parameters and functional problem
        #         cc = Control(cntrl)
        #         rf = ReducedFunctional(J, cc, derivative_cb_post = derivative_cb)
        #
        #         ccm  = moola.DolfinPrimalVector(cntrl)
        #
        #
        #         problem = MoolaOptimizationProblem(rf)
        #         solver = moola.BFGS(problem, ccm, options={'jtol': 1e-4,
        #                                                'gtol': 1e-9,
        #                                                'line_search_options' : {"ftol": 1e-4, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1},
        #                                                'maxiter': self.param['inv_options']['maxiter'],
        #                                                'mem_lim': 10})
        #
        #         sol = solver.solve()
        #         opt_var = sol['control']
        #         cntrl.vector().set_local(opt_var.array())


        #Scipy Optimization routine
        #opt_var = minimize(rf, method = 'L-BFGS-B', options = self.param['inv_options'])
        #map(lambda x: x[0].assign(x[1]), zip(cntrl,opt_var)) if type(cntrl) is list else cntrl.assign(opt_var)

        #Re-compute velocities with inversion results
        # reset()
        # clear_caches()
        # start_annotating()
        # self.solve_mom_eq()
        # stop_annotating()
        #
        # #Print out inversion results/parameter values
        # self.J_inv = self.comp_J_inv(verbose=True)

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
        eps_rp = self.param['eps_rp']

        eps = self.epsilon(U)
        exx = eps[0,0]
        eyy = eps[1,1]
        exy = eps[0,1]

        # Second invariant of the strain rate tensor squared
        eps_2 = (exx**2 + eyy**2 + exx*eyy + (exy)**2 + eps_rp**2)

        return eps_2

    def viscosity(self,U):
        B = self.rev_prmz(self.beta)
        n = self.param['n']

        eps_2 = self.effective_strain_rate(U)
        nu = 0.5 * B * eps_2**((1.0-n)/(2.0*n))

        return nu

    def comp_J_inv(self, verbose=False):

        u,v = split(self.U)

        u_obs = self.u_obs
        v_obs = self.v_obs
        u_std = self.u_std
        v_std = self.v_std

        alpha = self.alpha
        beta = self.beta
        beta_bgd = self.beta_bgd
        betadiff = beta-beta_bgd

        dIce = self.dIce
        dIce_gnd = self.dIce_gnd
        ds = self.ds
        nm = self.nm

        lambda_a = self.param['rc_inv'][0]
        delta_a = self.param['rc_inv'][1]
        delta_b = self.param['rc_inv'][2]
        gamma_a = self.param['rc_inv'][3]
        gamma_b = self.param['rc_inv'][4]

        J_ls = lambda_a*(u_std**(-2.0)*(u-u_obs)**2.0 + v_std**(-2.0)*(v-v_obs)**2.0)*self.dObs

        f = TrialFunction(self.Qp)
        f_alpha = Function(self.Qp)
        f_beta = Function(self.Qp)

        a = f*self.pTau*dIce
        L = (delta_a * alpha * self.pTau - gamma_a*inner(grad(alpha), grad(self.pTau)))*dIce
        solve(a == L, f_alpha )

        L = (delta_b * betadiff * self.pTau - gamma_b*inner(grad(betadiff), grad(self.pTau)))*dIce
        solve(a == L, f_beta )

        #
        # grad_alpha = grad(alpha)
        # grad_alpha_ = project(grad_alpha, self.RT)
        # lap_alpha = div(grad_alpha_)
        #
        #
        #
        # grad_betadiff = grad(betadiff)
        # grad_betadiff_ = project(grad_betadiff, self.RT)
        # lap_beta = div(grad_betadiff_)
        #
        # reg_a = delta_a * alpha - gamma_a*lap_alpha
        # reg_b = delta_b * betadiff - gamma_b*lap_beta
        #
        # J_reg_alpha = inner(reg_a,reg_a)*dIce_gnd
        # J_reg_beta = inner(reg_b,reg_b)*dIce

        J_reg_alpha = inner(f_alpha,f_alpha)*dIce
        J_reg_beta = inner(f_beta,f_beta)*dIce

        J = J_ls + J_reg_alpha + J_reg_beta

        return J


        if verbose:
            #Print out results
            J1 = assemble(J)
            J2 =  assemble(J_ls)
            J3 = assemble(J_reg_alpha)
            J4 = assemble(J_reg_beta)


            print('Inversion Details')
            print('delta_a: %.2e' % delta_a)
            print('delta_b: %.2e' % delta_b)
            print('gamma_a: %.2e' % gamma_a)
            print('gamma_b: %.2e' % gamma_b)
            print('J: %.2e' % J1)
            print('J_ls: %.2e' % J2)
            print('J_reg: %.2e' % sum([J3,J4]))
            print('J_reg_alpha: %.2e' % J3)
            print('J_reg_beta: %.2e' % J4)
            print('J_reg/J_cst: %.2e' % ((J3+J4)/(J2)))

    def comp_J_vaf(self, verbose=False):
        H = self.H_nps
        #B stands in for self.bed, which leads to a taping error
        B = Function(self.M)
        B.assign(self.bed, annotate=False)
        rhoi = Constant(self.param['rhoi'])
        rhow = Constant(self.param['rhow'])
        dIce = self.dIce
        dIce_gnd = self.dIce_gnd
        dt = self.dt

        b_ex = conditional(B < 0.0, 1.0, 0.0)
        HAF = b_ex * (H + rhow/rhoi*B) + (1-b_ex)*(H)
        J_vaf = HAF * dIce_gnd

        if verbose: print('J_vaf: {0}'.format(J_vaf))

        return J_vaf

    def comp_J_h2(self,verbose=False):

        J_h2 = self.H_np*self.H_np*self.dIce
        if verbose: print('J_h2: {0}'.format(J_h2))

        return J_h2


    def set_dQ_vaf(self, cntrl):
        J = self.timestep(adjoint_flag=1, qoi_func=self.comp_J_vaf)
        dJ = compute_gradient(J, cntrl)
        self.dQ_ts = dJ

    def set_dQ_h2(self, cntrl):
        J = self.timestep(adjoint_flag=1, qoi_func=self.comp_J_h2)
        dJ = compute_gradient(J, cntrl)
        self.dQ_ts = dJ

    # def set_dJ_inv(self, cntrl):
    #     J = Functional(self.J_inv)
    #     control = [Control(x) for x in cntrl] if type(cntrl) is list else Control(cntrl)
    #     dJ = compute_gradient(J, control, forget = False)
    #     self.dJ_inv = dJ


    def set_hessian_action(self, cntrl):
        if type(cntrl) is not list: cntrl = [cntrl]
        fopts = {'alpha': self.forward_alpha, 'beta': self.forward_beta, 'dual': self.forward_dual}
        forward = fopts['dual'] if len(cntrl) > 1 else fopts[cntrl[0].name()]

        reset()
        clear_caches()
        start_manager()
        J = forward(*cntrl)
        stop_manager()

        self.ddJ = SingleBlockHessian(J)

    def taylor_ver_inv(self,alpha_in):
        self.alpha = alpha_in
        self.def_mom_eq()
        self.solve_mom_eq()
        self.J_inv = self.comp_J_inv()
        return assemble(self.J_inv)

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

    # def taylor_ver_vaf(self,alpha_in, adjoint_flag=0):
    #     self.alpha = alpha_in
    #     self.H = self.H_init.copy(deepcopy=True)
    #     self.H_s = self.H_init.copy(deepcopy=True)
    #     self.H_np = self.H_init.copy(deepcopy=True)
    #     self.timestep(save=0, adjoint_flag=adjoint_flag)
    #     self.set_J_vaf()
    #     return assemble(self.J_vaf)


class ddJ_wrapper(object):
    def __init__(self, ddJ_action, cntrl):
        self.ddJ_action = ddJ_action
        self.ddJ_F = Function(cntrl.function_space())

    def apply(self,x):
        self.ddJ_F.vector().set_local(x.getArray())
        self.ddJ_F.vector().apply('insert')
        return self.ddJ_action(self.ddJ_F).vector().get_local()


# class MomentumSolver(EquationSolver):
#
#     def __init__(self, *args, **kwargs):
#         self.picard_params = kwargs.pop("picard_params", None)
#         self.J_p = kwargs.pop("J_p", None)
#         super(MomentumSolver, self).__init__(*args, **kwargs)
#
#     def forward_solve(self, x, deps):
#         #replace_map = dict(zip(self._EquationSolver__deps, deps))
#         replace_map = dict(list(zip(self.dependencies(), deps)))
#
#         if not self._EquationSolver__initial_guess is None:
#           x.assign(replace_map[self._EquationSolver__initial_guess])
#         replace_map[self.x()] = x
#
#         lhs = replace(self._EquationSolver__lhs, replace_map)
#         rhs = 0 if self._EquationSolver__rhs == 0 else replace(self._EquationSolver__rhs, replace_map)
#         J = replace(self._EquationSolver__J, replace_map)
#
#         solve(lhs == rhs, x, self._EquationSolver__bcs, J = replace(self.J_p, replace_map), form_compiler_parameters = self._EquationSolver__form_compiler_parameters, solver_parameters = self.picard_params)
#         solve(lhs == rhs, x, self._EquationSolver__bcs, J = J, form_compiler_parameters = self._EquationSolver__form_compiler_parameters, solver_parameters = self._EquationSolver__solver_parameters)
#
#         return



class MomentumSolver(EquationSolver):

    def __init__(self, *args, **kwargs):
        self.picard_params = kwargs.pop("picard_params", None)
        self.J_p = kwargs.pop("J_p", None)
        super(MomentumSolver, self).__init__(*args, **kwargs)

    def forward_solve(self, x, deps = None):
        if deps is None:
          deps = self.dependencies()
          replace_deps = lambda form : form
        else:
          from collections import OrderedDict
          replace_map = OrderedDict(zip(self.dependencies(), deps))
          replace_map[self.x()] = x
          replace_deps = lambda form : replace(form, replace_map)
        if not self._initial_guess_index is None:
          function_assign(x, deps[self._initial_guess_index])

        lhs = replace_deps(self._lhs)
        rhs = 0 if self._rhs == 0 else replace_deps(self._rhs)
        J_p = replace_deps(self.J_p)
        J = replace_deps(self._J)
        solve(lhs == rhs, x, self._bcs, J = self.J_p, form_compiler_parameters = self._form_compiler_parameters, solver_parameters = self.picard_params)
        solve(lhs == rhs, x, self._bcs, J = J, form_compiler_parameters = self._form_compiler_parameters, solver_parameters = self._solver_parameters)




def Hinit_gen(p_scale):
    def Hinit(x):
        """ Returns the primal representation. """
        #p_scale = [2.3e+08,10000.0]
        y = x.copy()
        for v,s in zip(y.vector_list,p_scale):
            v.data.vector().set_local(v.array()/s)

        if x.riesz_map.inner_product == "l2":
            return moola.DolfinPrimalVectorSet([vec.primal() for vec in y.vector_list],
            riesz_map = y.riesz_map)
        else:
            primal_vecs = zeros(len(y), dtype = "object")
            primal_vecs[:] = [v.primal() for v in y.vector_list]
            return moola.DolfinPrimalVectorSet(y.riesz_map.riesz_inv * primal_vecs,
            riesz_map = y.riesz_map)

    return Hinit
