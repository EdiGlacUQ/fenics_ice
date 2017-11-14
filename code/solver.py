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
            nu = self.viscosity(self.U)

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

            self.mom_F = (
                    #Membrance Stresses
                    -inner(grad(Phi_x), height * nu * as_vector([4 * u_x + 2 * v_y, u_y + v_x])) * self.dIce
                    - inner(grad(Phi_y), height * nu * as_vector([u_y + v_x, 4 * v_y + 2 * u_x])) * self.dIce

                    #Basal Drag
                    - inner(Phi, (1.0 - fl_ex) * B2 * as_vector([u,v])) * self.dIce

                    #Driving Stress
                    + ( div(Phi)*F - inner(grad(bed),W*Phi) ) * self.dIce

                    #Boundary condition
                    + inner(Phi * sigma_n, self.nm) * self.ds )

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
        solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.bcs, solver_parameters = self.param['solver_param'])
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
        B = exp(beta)

        gamma1 = self.param['gamma1']
        gamma2 = self.param['gamma2']

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

        J_ls = 0.0*(u_std**(-2)*(u-u_obs)**2 + v_std**(-2)*(v-v_obs)**2)*self.dObs
        J_log = ((V**2)*ln( ((u**2 + v**2)**2 + eta) /  ((u_obs**2 + v_obs**2)**2 + eta) )**2) *self.dObs
        J_reg_alpha = gamma1*inner(grad(exp(alpha)),grad(exp(alpha)))*self.dIce_gnd
        J_reg_beta = 0.0*gamma2*(inner(beta - beta_bgd,beta - beta_bgd)*self.dIce_flt +
                inner(beta - beta_bgd,beta - beta_bgd)*self.dIce_gnd)

        J = Functional(J_ls + J_log+ J_reg_alpha + J_reg_beta)

        #control = [Control(alpha), Control(beta)]
        control = Control(alpha)

        rf = ReducedFunctional(J, control, derivative_cb_post = derivative_cb)

        #Optimization routine
        opt_var = minimize(rf, method = 'L-BFGS-B', options = self.param['inv_options'])

        #self.alpha.assign(opt_var[0])
        #self.beta.assign(opt_var[1])
        self.alpha.assign(opt_var)

        #Compute velocities with inverted basal drag
        self.solve_mom_eq()

        #Print out results
        J1 =  assemble(J_ls)
        J2 =  assemble(J_log)
        J3 = assemble(J_reg_alpha)
        J4 = assemble(J_reg_beta)


        print 'Inversion Details'
        print 'J: %.2e' % sum([J1,J2,J3,J4])
        print 'gamma1: %.2e' % gamma1
        print 'gamma2: %.2e' % gamma2
        print 'J_cst: %.2e' % sum([J1,J2])
        print 'J_ls: %.2e' % J1
        print 'J_log: %.2e' % J2
        print 'J_reg: %.2e' % sum([J3,J4])
        print 'J_reg_alpha: %.2e' % J3
        print 'J_reg_beta: %.2e' % J4
        print 'J_reg/J_cst: %.2e' % ((J2+J3)/(J1+J2))

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
