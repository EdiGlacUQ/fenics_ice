from dolfin import *
import numpy as np
import timeit
from IPython import embed
import matplotlib.pyplot as plt


class ssa_solver:

    def __init__(self, model):
        self.model = model

        #Fields
        self.surf = model.surf
        self.bed = model.bed
        self.height = model.thick
        self.mask = model.mask
        self.B2 = model.bdrag
        self.bmelt = model.bmelt
        self.nm = model.nm

        #Constants
        self.rhoi = model.rhoi
        self.rhow = model.rhow
        self.delta = model.delta
        self.g = model.g
        self.n = model.n
        self.eps_rp = model.eps_rp
        self.A = model.A

        #parameters
        self.eq_def = model.eq_def
        self.solve_param = model.solve_param

        #Function Spaces
        self.V = model.V
        self.Q = model.Q
        self.M = model.M

        #Functions
        self.U = model.U
        self.dU = model.dU
        self.Phi = model.Phi

        #Cells
        self.cf = model.cf
        self.OMEGA_X = model.OMEGA_X
        self.OMEGA_ICE = model.OMEGA_ICE

        #Facets
        self.ff = model.ff
        self.GAMMA_DEF = model.GAMMA_DEF
        self.GAMMA_LAT = model.GAMMA_LAT

        #Measures
        self.dx = Measure('dx', domain=model.mesh, subdomain_data=self.cf)
        self.dIce = self.dx(self.OMEGA_ICE)

    def def_mom_eq(self):
        surf = self.surf
        bed = self.bed
        height = self.height
        mask = self.mask
        B2 = self.B2
        rhoi = self.rhoi
        rhow = self.rhow
        delta = self.delta
        g = self.g
        n = self.n
        A = self.A
        dIce = self.dIce




        #Equations from Action Principle [Dukowicz et al., 2010, JGlac, Eq 94]
        if self.eq_def == 1:

            #Driving Stress: Simple
            #Simple version
            #gradS = grad(surf)
            #tau_drv = project(rhoi*g*height*gradS, model.V)
            #Ds = dot(tau_drv, model.U)

            #Driving Stress
            height_s = -rhow/rhoi * bed

            F = conditional(gt(height,height_s),0.5*rhoi*g*height**2,
                0.5*rhoi*g*(delta*height**2 + (1-delta)*height_s**2))

            W = conditional(gt(height,height_s), rhoi*g*height, rhoi*g*height_s)

            Ds = dot(self.U, grad(F) + W*grad(bed))

            #Terminating margin boundary condition
            fl_ex = conditional(height <= height_s, 1.0, 0.0)

            #Bottom of ice sheet, either bed or draft
            R_f = ((1.0 - fl_ex) * bed
               + (fl_ex) * (-rhoi / rhow) * height)

            draft = Min(R_f,0)

            sigma_n = 0.5 * rhoi * g * ((height ** 2) - (rhow / rhoi) * (draft ** 2))
            mgn_bc = inner(self.U("+") * sigma_n("+"), self.nm("+")) * abs(jump(mask))

            #Viscous Dissipation
            epsdot = self.effective_strain_rate(self.U)
            Vd = (2.0*n)/(n+1.0) * (A**(-1.0/n)) * (epsdot**((n+1.0)/(2.0*n)))

            #Sliding law
            fl_ex = conditional(height <= height_s, 1.0, 0.0)
            Sl = 0.5 * (1.0 -fl_ex) * B2 * dot(self.U,self.U)

            # action :
            Action = (height*Vd + Ds + Sl)*dIce + mgn_bc*dS

            # the first variation of the action in the direction of a
            # test function ; the extremum :
            self.mom_F = derivative(Action, self.U)

            # the first variation of the extremum in the direction
            # a trial function ; the Jacobian :
            self.mom_Jac = derivative(self.mom_F, self.U)



        #Equations in weak form
        else:
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

            #Switch parameters
            height_s = -rhow/rhoi * bed
            fl_ex = conditional(height <= height_s, 1.0, 0.0)

            #Bottom of ice sheet, either bed or draft
            R_f = ((1.0 - fl_ex) * bed
               + (fl_ex) * (-rhoi / rhow) * height)

            draft = Min(R_f,0)

            #Ice Sheet Surface
            s = R_f + height

            #Terminating margin boundary condition
            sigma_n = 0.5 * rhoi * g * ((height ** 2) - (rhow / rhoi) * (draft ** 2))
            self.mom_F = ( -inner(grad(Phi_x), height * nu * as_vector([4 * u_x + 2 * v_y, u_y + v_x])) * dIce
                    - inner(grad(Phi_y), height * nu * as_vector([u_y + v_x, 4 * v_y + 2 * u_x])) * dIce
                    - inner(Phi, (1.0 - fl_ex) * B2 * as_vector([u,v])) * dIce
                    - inner(Phi, rhoi * g * height * grad(s)) * dIce
                    + inner(Phi("+")*sigma_n,self.nm("+"))*np.abs(jump(mask)) * dS)

            self.mom_Jac = derivative(self.mom_F, self.U)


    def solve_mom_eq(self):

        #Dirichlet Boundary Conditons at lateral domain margins
        self.bc_dmn = [DirichletBC(self.V, (0.0, 0.0), self.ff, self.GAMMA_LAT)]

        solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.bc_dmn,solver_parameters = self.solve_param)


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

        eps = self.epsilon(U)
        exx = eps[0,0]
        eyy = eps[1,1]
        exy = eps[0,1]

        # Second invariant of the strain rate tensor squared
        eps_2 = (exx**2 + eyy**2 + exx*eyy + (exy)**2 + self.eps_rp**2)

        return eps_2

    def viscosity(self,U):
        A = self.A
        n = self.n

        eps_2 = self.effective_strain_rate(U)
        nu = 0.5 * A**(-1.0/n) * eps_2**((1.0-n)/(2.0*n))

        return nu
