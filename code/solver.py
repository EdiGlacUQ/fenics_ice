from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint_sqrt_masslump import *

import matplotlib.pyplot as plt
import numpy as np
import ufl

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
        self.M = model.M
        self.RT = model.RT

        #Trial/Test Functions
        self.U = Function(self.V, name = "U")
        self.U_np = Function(self.V, name = "U_np")
        self.Phi = TestFunction(self.V)
        self.Ksi = TestFunction(self.M)

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
        B2 = alpha*alpha

        #Switch parameters
        H_s = -rhow/rhoi * bed
        fl_ex = ufl.operators.Conditional(H <= H_s, Constant(1.0), Constant(0.0))
        #fl_ex = Constant(0.0)

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


    def solve_mom_eq(self, annotate_flag=True):
        #Dirichlet Boundary Conditons: Zero flow

        bc0 = DirichletBC(self.V, self.latbc, self.ff, self.GAMMA_LAT)
        bc1 = DirichletBC(self.V, (0.0, 0.0), self.ff, self.GAMMA_NF)
        self.bcs = [bc0, bc1]


        #Non zero initial perturbation
        #self.init_guess()
        #parameters['krylov_solver']['nonzero_initial_guess'] = True
        t0 = time.time()
        picard_params = {"nonlinear_solver":"newton",
                         "newton_solver":{"linear_solver":"umfpack",
                                          "maximum_iterations":200,
                                          "absolute_tolerance":1.0e-8,
                                          "relative_tolerance":5.0e-2,
                                          "convergence_criterion":"incremental",
                                          "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}
        newton_params = {"nonlinear_solver":"newton",
                         "newton_solver":{"linear_solver":"umfpack",
                                          "maximum_iterations":20,
                                          "absolute_tolerance":1.0e-9,
                                          "relative_tolerance":1.0e-9,
                                          "convergence_criterion":"incremental",
                                          "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}


        from dolfin_adjoint_custom import EquationSolver
        J_p = self.mom_Jac_p


        class MomentumSolver(EquationSolver):
          def forward_solve(self, x, deps):
            #replace_map = dict(zip(self._EquationSolver__deps, deps))
            replace_map = dict(zip(self.dependencies(), deps))

            if not self._EquationSolver__initial_guess is None:
              x.assign(replace_map[self._EquationSolver__initial_guess])
            replace_map[self.x()] = x

            lhs = replace(self._EquationSolver__lhs, replace_map)
            rhs = 0 if self._EquationSolver__rhs == 0 else replace(self._EquationSolver__rhs, replace_map)
            J = replace(self._EquationSolver__J, replace_map)

            solve(lhs == rhs, x, self._EquationSolver__bcs, J = replace(J_p, replace_map), form_compiler_parameters = self._EquationSolver__form_compiler_parameters, solver_parameters = picard_params)
            solve(lhs == rhs, x, self._EquationSolver__bcs, J = J, form_compiler_parameters = self._EquationSolver__form_compiler_parameters, solver_parameters = self._EquationSolver__solver_parameters)

            return

        MomentumSolver(self.mom_F == 0, self.U, bcs = self.bcs, solver_parameters = newton_params).solve(annotate=annotate_flag)#self.param['solver_param'])
        t1 = time.time()
        print "Time for solve: ", t1-t0


    def def_thickadv_eq(self):
        U = self.U
        U_np = self.U_np
        Ksi = self.Ksi
        trial_H = self.trial_H
        H_np = self.H_np
        H_s = self.H_s
        H = self.H
        dt = self.dt
        nm = self.nm
        dIce = self.dIce
        ds = self.ds
        dS = self.dS

        self.thickadv = (inner(Ksi, ((trial_H - H_np) / dt)) * dIce
        - inner(grad(Ksi), U_np * 0.5 * (trial_H + H_np)) * dIce
        + inner(jump(Ksi), jump(0.5 * (dot(U_np, nm) + abs(dot(U_np, nm))) * 0.5 * (trial_H + H_np))) * dS
        + inner(Ksi, dot(U_np * 0.5 * (trial_H + H_np), nm)) * ds)

        self.thickadv_split = replace(self.thickadv, {U_np:0.5 * (self.U + self.U_np)})

        bc0 = DirichletBC(self.M, self.H, self.ff, self.GAMMA_LAT)
        bc1 = DirichletBC(self.M, (0.0), self.ff, self.GAMMA_TMN)
        self.H_bcs = [bc0, bc1]

    def solve_thickadv_eq(self):

        H_s = self.H_s
        a, L = lhs(self.thickadv), rhs(self.thickadv)
        solve(a==L,H_s,bcs = self.H_bcs)

    def solve_thickadv_split_eq(self):
        H_nps = self.H_nps
        a, L = lhs(self.thickadv_split), rhs(self.thickadv_split)
        solve(a==L,H_nps, bcs = self.H_bcs)

    def timestep(self, save = 1):
        U = self.U
        U_np = self.U_np
        H = self.H
        H_np = self.H_np
        H_s = self.H_s
        H_nps = self.H_nps

        n_steps = self.param['n_steps']
        dt = self.dt

        if save:
            output_hts = File("H_ts.pvd", "compressed")
            output_uts = File("U_ts.pvd", "compressed")


        self.def_thickadv_eq()
        self.def_mom_eq()
        self.solve_mom_eq()
        U_np.assign(U)

        t=0.0

        adj_start_timestep()
        for n in xrange(n_steps):
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

            #Record
            if save:
                output_hts << (H_np, t)
                output_uts << (U_np, t)

            end()
            adj_inc_timestep()

    def inversion(self):

        #Record value of functional during minimization
        self.F_iter = 0
        self.F_vals = np.zeros(10*self.param['inv_options']['maxiter']); #Initialize array

        #Callback function during minimization storing cost function value
        def derivative_cb(j, dj, m):
            self.F_vals[self.F_iter] = j
            self.F_iter += 1
            print "j = %f" % (j)

        #Initial equation definition and solve
        self.def_mom_eq()
        self.solve_mom_eq()

        #Set up cost functional
        self.set_J_inv()
        J = Functional(self.J_inv)

        #Control parameters and functional problem
        control = Control(self.alpha)
        #control = [Control(self.alpha), Control(self.beta)]
        rf = ReducedFunctional(J, control, derivative_cb_post = derivative_cb)

        #Optimization routine
        opt_var = minimize(rf, method = 'L-BFGS-B', options = self.param['inv_options'])

        #Save results
        self.alpha.assign(opt_var)
        #self.alpha.assign(opt_var[0])
        #self.beta.assign(opt_var[1])

        #Re-compute velocities with inversion results
        adj_reset()
        parameters["adjoint"]["stop_annotating"] = False
        self.solve_mom_eq()
        parameters["adjoint"]["stop_annotating"] = True

        #Print out inversion results/parameter values
        self.set_J_inv(verbose = True)

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
        B = exp(self.beta)
        n = self.param['n']

        eps_2 = self.effective_strain_rate(U)
        nu = 0.5 * B * eps_2**((1.0-n)/(2.0*n))

        return nu

    def init_guess(self):
        #Simplify accessing fields and parameters
        bed = self.bed
        H = self.H
        alpha = self.alpha

        rhoi = self.param['rhoi']
        rhow = self.param['rhow']
        delta = 1.0 - rhoi/rhow
        g = self.param['g']
        n = self.param['n']
        tol = self.param['tol']

        B2 = exp(alpha)
        H_s = -rhow/rhoi * bed
        fl_ex = conditional(H <= H_s, 1.0, 0.0)

        s = project((1-fl_ex) * (bed + H),self.Q)
        grads = as_vector([s.dx(0), s.dx(1)])
        U_ = project((1-fl_ex)*(rhoi*g*H*grads)/B2, self.V)

        self.U.assign(U_)

        vtkfile = File('U_init.pvd')
        vtkfile << self.U

    def set_J_inv(self, verbose=False):

        u, v = split(self.U)
        u_obs = self.u_obs
        v_obs = self.v_obs
        u_std = self.u_std
        v_std = self.v_std

        alpha = self.alpha
        beta = self.beta
        beta_bgd = self.beta_bgd

        dIce = self.dIce
        ds = self.ds
        nm = self.nm

        J_ls = (u_std**(-2)*(u-u_obs)**2 + v_std**(-2)*(v-v_obs)**2)*self.dObs

        lambda_a = self.param['rc_inv'][0]
        lambda_b = self.param['rc_inv'][1]
        delta_a = self.param['rc_inv'][2]
        delta_b = self.param['rc_inv'][3]

        grad_alpha = grad(alpha)
        grad_alpha_ = project(grad_alpha, self.RT)
        lap_alpha = div(grad_alpha_)

        betadiff = beta-beta_bgd
        grad_betadiff = grad(betadiff)
        grad_betadiff_ = project(grad_betadiff, self.RT)
        lap_beta = div(grad_betadiff_)

        reg_a = lambda_a * alpha - delta_a*lap_alpha
        reg_a_bndry = delta_a*inner(grad_alpha,nm)

        reg_b = lambda_b * betadiff - delta_b*lap_beta
        reg_b_bndry = delta_b*inner(grad_betadiff,nm)

        J_reg_alpha = inner(reg_a,reg_a)*dIce + inner(reg_a_bndry,reg_a_bndry)*ds
        J_reg_beta = inner(reg_b,reg_b)*dIce + inner(reg_b_bndry,reg_b_bndry)*ds

        J = J_ls + J_reg_alpha + J_reg_beta

        self.J_inv = J



        if verbose:
            #Print out results
            J1 = assemble(J)
            J2 =  assemble(J_ls)
            J3 = assemble(J_reg_alpha)
            J4 = assemble(J_reg_beta)


            print 'Inversion Details'
            print 'lambda_a: %.2e' % lambda_a
            print 'lambda_b: %.2e' % lambda_b
            print 'delta_a: %.2e' % delta_a
            print 'delta_b: %.2e' % delta_b
            print 'J: %.2e' % J1
            print 'J_ls: %.2e' % J2
            print 'J_reg: %.2e' % sum([J3,J4])
            print 'J_reg_alpha: %.2e' % J3
            print 'J_reg_beta: %.2e' % J4
            print 'J_reg/J_cst: %.2e' % ((J3+J4)/(J2))

    def set_J_vaf(self, verbose=False):
        H = self.H
        B = Function(self.M)
        B.assign(self.bed, annotate=False)
        rhoi = self.param['rhoi']
        rhow = self.param['rhow']
        dIce = self.dIce
        dIce_gnd = self.dIce_gnd
        dt = self.dt

        b_ex = conditional(B < 0.0, 1.0, 0.0)
        HAF = b_ex * (H + rhow/rhoi*B) + (1-b_ex)*(H)

        self.J_vaf = HAF * dIce_gnd

    def comp_dJ_vaf(self, cntrl):
        J = Functional(self.J_vaf)
        control = Control(cntrl)
        dJ = compute_gradient(J, control, forget = False)
        self.dJ_vaf = dJ


    def set_hessian_action(self, cntrl):
        J = Functional(self.J_inv)
        cc = Control(cntrl)
        self.hess = hessian(J,cc)

    def taylor_ver(self,alpha_in, annotate_flag=False):
        self.alpha = project(alpha_in, self.Q, annotate=annotate_flag)
        self.def_mom_eq()
        #self.solve_mom_eq(annotate_flag=False) #Makes no difference for Hessian
        self.solve_mom_eq()
        self.J =  inner(self.U,self.U)*self.dIce
        #self.J =  inner(self.alpha**2,self.alpha**2)*self.dIce
        return assemble(self.J)

    def taylor_ver2(self,alpha_in):
        #Same as above for a single test
        self.alpha = alpha_in
        self.def_mom_eq()
        self.solve_mom_eq()
        self.J =  inner(self.U,self.U)*self.dIce
        return assemble(self.J)









# class domain_x(SubDomain):
#     def set_mask(self,fn):
#         self.mask = fn.copy(deepcopy=True)
#         self.cntr = 0
#         self.xx = []
#         self.yy = []
#
#     def inside(self,x, on_boundary):
#         tol = 1e-2
#         mask = self.mask
#
#         CC = mask(x[0],x[1])
#
#         try:
#             CE = mask(x[0] + tol,x[1])
#         except:
#             CE = 0.0
#         try:
#             CW = mask(x[0] - tol ,x[1])
#         except:
#             CW = 0.0
#         try:
#             CN = mask(x[0], x[1] + tol)
#         except:
#             CN = 0.0
#         try:
#             CS = mask(x[0], x[1] - tol)
#         except:
#             CS = 0.0
#
#         mv = np.max([CC, CE, CW, CN, CS])
#         if near(mv,0.0,tol):
#             self.cntr += 1
#             self.xx.append(x[0])
#             self.yy.append(x[1])
#             return True
#         else:
#             return False



#Small perturbation so derivative of regularization is not zero for uniform initial guess
#pert = Function(self.Q2, name = "pert")
#noise_ = 0.005

# max_ = alpha.vector().norm("linf")
# pert.vector().set_local(max_*noise_*np.random.uniform(0,1,self.alpha.vector().array().size))
# pert.vector().apply("insert")
# alpha.vector().axpy(1.0,pert.vector())
#
# max_ = beta.vector().norm("linf")
# pert.vector().set_local(max_*noise_*np.random.uniform(0,1,self.beta.vector().array().size))
# pert.vector().apply("insert")
# beta.vector().axpy(1.0,pert.vector())


#Classical Regularization
# B2 = exp(alpha)
# Aglen = exp(beta)
# Aglen_bgd = exp(beta_bgd)
# J_ls = gc1*(u_std**(-2)*(u-u_obs)**2 + v_std**(-2)*(v-v_obs)**2)*self.dObs
# J_reg_alpha = gc1*inner(B2 + grad(B2),B2 + grad(B2))*self.dIce
# J_reg_beta = gc2*inner(beta - beta_bgd,beta - beta_bgd)*self.dIce_gnd
# J = Functional(J_ls + J_reg_alpha + J_reg_beta + J_reg2_beta)
# J0 = assemble(J_ls + J_reg_alpha + J_reg_beta + J_reg2_beta)
