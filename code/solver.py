from dolfin import *
from dolfin_adjoint import *

import matplotlib.pyplot as plt
import numpy as np

import timeit
import time
from IPython import embed



class ssa_solver:

    def __init__(self, model):
        # Enable aggressive compiler options
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math -march=native"
        parameters["form_compiler"]["optimize"] = False

        self.model = model
        self.param = model.param

        #Fields
        self.bed = model.bed
        self.height = model.thick
        self.surf = model.surf
        self.beta = model.beta
        self.mask = model.mask
        self.alpha = model.alpha
        self.bmelt = model.bmelt
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

        #Trial/Test Functions
        self.U = Function(self.V)
        self.dU = TrialFunction(self.V)
        self.Phi = TestFunction(self.V)

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



    def def_mom_eq(self):

        #Simplify accessing fields and parameters
        bed = self.bed
        height = self.height
        surf = self.surf
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


        #Equations from Action Principle [Dukowicz et al., 2010, JGlac, Eq 94]
        if self.param['eq_def'] == 'action':

            #Driving Stress
            height_s = -rhow/rhoi * bed

            F = conditional(gt(height,height_s),0.5*rhoi*g*height**2,
                0.5*rhoi*g*(delta*height**2 + (1-delta)*height_s**2))

            W = conditional(gt(height,height_s), rhoi*g*height, rhoi*g*height_s)

            Ds = dot(self.U, grad(F) + W*grad(bed))

            fl_ex = conditional(height <= height_s, 1.0, 0.0)


            #bottom of ice sheet, either bed or draft
            R_f = ((1.0 - fl_ex) * bed
               + (fl_ex) * (-rhoi / rhow) * height)

            draft = Min(R_f,0)

            #Terminating margin boundary condition
            ii = project(conditional(mask > 0.0, 1.0, 0.0),self.M, annotate=False)
            sigma_n = 0.5 * rhoi * g * ((height ** 2) - (rhow / rhoi) * (draft ** 2))
            #mgn_bc = inner(self.U("+") * sigma_n("+"), self.nm("+"))*abs(jump(ii))
            mgn_bc = inner(self.U("+") * sigma_n("+"), self.nm("+")) * abs(jump(mask))

            #Viscous Dissipation
            epsdot = self.effective_strain_rate(self.U)
            Vd = (2.0*n)/(n+1.0) * (A**(-1.0/n)) * (epsdot**((n+1.0)/(2.0*n)))

            #Sliding law
            B2 = exp(alpha)
            fl_ex = conditional(height <= height_s, 1.0, 0.0)
            Sl = 0.5 * (1.0 -fl_ex) * B2 * dot(self.U,self.U)

            # action :
            Action = (height*Vd + Ds + Sl)*dIce - mgn_bc*dS

            # the first variation of the action in the direction of a
            # test function ; the extremum :
            self.mom_F = derivative(Action, self.U)

            # the first variation of the extremum in the direction
            # a trial function ; the Jacobian :
            self.mom_Jac = derivative(self.mom_F, self.U)

        #Equations in weak form
        else:

            if self.param['eq_def'] != 'weak':
                print 'Unrecognized eq_def, resorting to weak form'

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
            B2 = exp(alpha)

            #Switch parameters
            height_s = -rhow/rhoi * bed
            fl_ex = conditional(height <= height_s, 1.0, 0.0)

            #Driving stress quantities
            F = (1 - fl_ex) * 0.5*rhoi*g*height**2 + \
                (fl_ex) * 0.5*rhoi*g*(delta*height**2 + (1-delta)*height_s**2 )

            W = (1 - fl_ex) * rhoi*g*height + \
                (fl_ex) * rhoi*g*height_s

            draft = (fl_ex) * (rhoi / rhow) * height

            #Terminating margin boundary condition
            sigma_n = 0.5 * rhoi * g * ((height ** 2) - (rhow / rhoi) * (draft ** 2)) - F
            sigma_n2 = 0.5 * rhoi * g * ((height ** 2) - (rhow / rhoi) * (draft ** 2))

            embed()
            self.mom_F = (
                    #Membrance Stresses
                    -inner(grad(Phi_x), height * nu * as_vector([4 * u_x + 2 * v_y, u_y + v_x])) * self.dIce
                    - inner(grad(Phi_y), height * nu * as_vector([u_y + v_x, 4 * v_y + 2 * u_x])) * self.dIce

                    #Basal Drag
                    - inner(Phi, (1.0 - fl_ex) * B2 * as_vector([u,v])) * self.dIce

                    #Driving Stress
                    #+ ( div(Phi)*F - inner(grad(bed),W*Phi) ) * self.dIce
                    - inner(Phi, rhoi * g * height * grad(surf)) * dIce

                    #Boundary condition
                    #+ inner(Phi * sigma_n, self.nm) * self.ds )
                    + inner(Phi * sigma_n2, self.nm) * self.ds )

            self.mom_Jac_p = replace(derivative(self.mom_F, self.U), {U_marker:self.U})
            self.mom_F = replace(self.mom_F, {U_marker:self.U})
            self.mom_Jac = derivative(self.mom_F, self.U)


    def solve_mom_eq(self):
        #Dirichlet Boundary Conditons: Zero flow
        bc0 = DirichletBC(self.V, (0.0, 0.0), self.ff, self.GAMMA_LAT)
        bc1 = DirichletBC(self.V, (0.0, 0.0), self.ff, self.GAMMA_NF)
        self.bcs = [bc0, bc1]

        #Non zero initial perturbation
        #self.init_guess()
        #parameters['krylov_solver']['nonzero_initial_guess'] = True
        t0 = time.time()
        picard_params = {"nonlinear_solver":"newton",
                         "newton_solver":{"linear_solver":"umfpack",
                                          "maximum_iterations":200,
                                          "absolute_tolerance":1.0e-16,
                                          "relative_tolerance":5.0e-2,
                                          "convergence_criterion":"incremental",
                                          "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}
        newton_params = {"nonlinear_solver":"newton",
                         "newton_solver":{"linear_solver":"umfpack",
                                          "maximum_iterations":20,
                                          "absolute_tolerance":1.0e-16,
                                          "relative_tolerance":1.0e-9,
                                          "convergence_criterion":"incremental",
                                          "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}
        from dolfin_adjoint_custom import EquationSolver
        J_p = self.mom_Jac_p

        class MomentumSolver(EquationSolver):
          def forward_solve(self, x, deps):
            replace_map = dict(zip(self._EquationSolver__deps, deps))

            if not self._EquationSolver__initial_guess is None:
              x.assign(replace_map[self._EquationSolver__initial_guess])
            replace_map[self.x()] = x

            lhs = replace(self._EquationSolver__lhs, replace_map)
            rhs = 0 if self._EquationSolver__rhs == 0 else replace(self._EquationSolver__rhs, replace_map)
            J = replace(self._EquationSolver__J, replace_map)

            solve(lhs == rhs, x, self._EquationSolver__bcs, J = replace(J_p, replace_map), form_compiler_parameters = self._EquationSolver__form_compiler_parameters, solver_parameters = picard_params)
            solve(lhs == rhs, x, self._EquationSolver__bcs, J = J, form_compiler_parameters = self._EquationSolver__form_compiler_parameters, solver_parameters = self._EquationSolver__solver_parameters)

            return

        MomentumSolver(self.mom_F == 0, self.U, bcs = self.bcs, solver_parameters = newton_params).solve()#self.param['solver_param'])
        t1 = time.time()
        print "Time for solve: ", t1-t0

    def inversion(self):

        u_obs = self.u_obs
        v_obs = self.v_obs
        u_std = self.u_std
        v_std = self.v_std

        alpha = self.alpha
        beta = self.beta
        beta_bgd = beta.copy(deepcopy=True)

        #Small perturbation so derivative of regularization is not zero
        pertf = np.random.uniform(0.98,1.02,self.alpha.vector().array().size)

        alpha.vector().set_local(np.multiply(pertf,alpha.vector()[:].array()))
        alpha.vector().apply("insert")

        beta.vector().set_local(np.multiply(pertf,beta.vector()[:].array()))
        beta.vector().apply("insert")

        gc1 = self.param['gc1']
        gc2 = self.param['gc2']
        gr1 = self.param['gr1']
        gr2 = self.param['gr2']
        gr3 = self.param['gr3']

        #Record value of functional during minimization
        self.F_iter = 0
        self.F_vals = np.zeros(10*self.param['inv_options']['maxiter']);

        def derivative_cb(j, dj, m):
            self.F_vals[self.F_iter] = j
            self.F_iter += 1
            print "j = %f" % (j)

        #Initial equation definition and solve
        self.def_mom_eq()
        self.solve_mom_eq()

        #Inversion Code
        u, v = split(self.U)

        #Define functional and control variable
        V = 50 #Velocity scale for Non-Dimensionalization
        eta = 5 #Minimium velocity

        J_ls = gc1*(u_std**(-2)*(u-u_obs)**2 + v_std**(-2)*(v-v_obs)**2)*self.dObs
        J_log = gc2*((V**2)*ln( ((u**2 + v**2)**2 + eta) /  ((u_obs**2 + v_obs**2)**2 + eta) )**2) *self.dObs
        J_reg_alpha = gr1*inner(grad(exp(alpha)),grad(exp(alpha)))*self.dIce_gnd
        J_reg_beta = gr2*inner(beta - beta_bgd,beta - beta_bgd)*self.dIce_gnd
        J_reg2_beta = gr3*inner(grad(exp(beta)),grad(exp(beta)))*self.dIce

        J = Functional(J_ls + J_log+ J_reg_alpha + J_reg_beta + J_reg2_beta)

        control = [Control(alpha), Control(beta)]
        #control = Control(alpha)

        rf = ReducedFunctional(J, control, derivative_cb_post = derivative_cb)

        #Optimization routine
        opt_var = minimize(rf, method = 'L-BFGS-B', options = self.param['inv_options'])

        self.alpha.assign(opt_var[0])
        self.beta.assign(opt_var[1])
        #self.alpha.assign(opt_var)

        #Compute velocities with inverted basal drag
        self.solve_mom_eq()

        #Print out results
        J1 =  assemble(J_ls)
        J2 =  assemble(J_log)
        J3 = assemble(J_reg_alpha)
        J4 = assemble(J_reg_beta)
        J5 = assemble(J_reg2_beta)


        print 'Inversion Details'
        print 'J: %.2e' % sum([J1,J2,J3,J4])
        print 'gc1: %.2e' % gc1
        print 'gc2: %.2e' % gc2
        print 'gr1: %.2e' % gr1
        print 'gr2: %.2e' % gr2
        print 'gr3: %.2e' % gr3
        print 'J_cst: %.2e' % sum([J1,J2])
        print 'J_ls: %.2e' % J1
        print 'J_log: %.2e' % J2
        print 'J_reg: %.2e' % sum([J3,J4,J5])
        print 'J_reg_alpha: %.2e' % J3
        print 'J_reg_beta: %.2e' % J4
        print 'J_reg2_beta: %.2e' % J5
        print 'J_reg/J_cst: %.2e' % ((J3+J4+J5)/(J1+J2))

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
        height = self.height
        alpha = self.alpha

        rhoi = self.param['rhoi']
        rhow = self.param['rhow']
        delta = 1.0 - rhoi/rhow
        g = self.param['g']
        n = self.param['n']
        tol = self.param['tol']

        B2 = exp(alpha)
        height_s = -rhow/rhoi * bed
        fl_ex = conditional(height <= height_s, 1.0, 0.0)

        s = project((1-fl_ex) * (bed + height),self.Q)
        grads = as_vector([s.dx(0), s.dx(1)])
        U_ = project((1-fl_ex)*(rhoi*g*height*grads)/B2, self.V)

        self.U.assign(U_)

        vtkfile = File('U_init.pvd')
        vtkfile << self.U



class domain_x(SubDomain):
    def set_mask(self,fn):
        self.mask = fn.copy(deepcopy=True)
        self.cntr = 0
        self.xx = []
        self.yy = []

    def inside(self,x, on_boundary):
        tol = 1e-2
        mask = self.mask

        CC = mask(x[0],x[1])

        try:
            CE = mask(x[0] + tol,x[1])
        except:
            CE = 0.0
        try:
            CW = mask(x[0] - tol ,x[1])
        except:
            CW = 0.0
        try:
            CN = mask(x[0], x[1] + tol)
        except:
            CN = 0.0
        try:
            CS = mask(x[0], x[1] - tol)
        except:
            CS = 0.0

        mv = np.max([CC, CE, CW, CN, CS])
        if near(mv,0.0,tol):
            self.cntr += 1
            self.xx.append(x[0])
            self.yy.append(x[1])
            return True
        else:
            return False
