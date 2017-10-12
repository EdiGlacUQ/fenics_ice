from dolfin import *
import numpy as np
import timeit
from IPython import embed
from numpy.random import rand
import matplotlib.pyplot as plt


class model:

    def __init__(self, mesh_in, param_in):

        #Default Mesh/Function Spaces
        self.mesh = Mesh(mesh_in)
        self.nm = FacetNormal(self.mesh)
        self.V = VectorFunctionSpace(self.mesh,'Lagrange',1,dim=2)
        self.Q = FunctionSpace(self.mesh,'Lagrange',1)
        self.M = FunctionSpace(self.mesh,'DG',0)

        #Initiate Functions
        self.U = Function(self.V)
        self.dU = TrialFunction(self.V)
        self.Phi = TestFunction(self.V)

        #Initiate parameters
        self.init_param(param_in)

    def init_param(self, param_in):

        #Constants for ice sheet modelling
        param = {}
        param['ty'] = 365*24*60*60  #seconds in year
        param['rhoi'] =  917.0      #density ice
        param['rhow'] =   1000.0     #density water
        param['g'] =  9.81           #acceleration due to gravity
        param['n'] =  3.0            #glen's flow law exponent
        param['eps_rp'] =  1e-5      #effective strain regularization
        param['A'] =  10**(-16)      #Creep paramater
        param['tol'] =  1e-6         #Tolerance for tests
        param['gamma'] =  1.0          #Cost function scaling parameter

        #Output
        param['outdir'] = './output/'

        #Equation and solver
        param['eq_def'] = 'action'
        param['solver'] = 'default'
        param['solver_param'] = {}

        #Solver options
        param['snes_linesearch_alpha'] = 1e-6
        param['solver_petsc'] = {'nonlinear_solver'      : 'snes',
                            'snes_solver':
                            {
                            'linear_solver'         : 'cg',
                            'preconditioner'        : 'jacobi',
                            'line_search'           : 'bt',
                            'relative_tolerance'    : 1e-15,
                            'absolute_tolerance'    : 1e-15,
                            'solution_tolerance'    : 1e-15
                            }}

        #Default fenics solver. No line search.
        param['solver_default']= {'newton_solver' :
                {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'jacobi',
                'relative_tolerance'       : 1e-15,
                'relaxation_parameter'     : 0.7,
                'absolute_tolerance'       : 10.0,
                'maximum_iterations'       : 50,
                'error_on_nonconvergence'  : False,
                }}


        param['inv_options'] = {'disp': True, 'maxiter': 15, 'factr': 0.0}

        #Update default values based on input
        param.update(param_in)

        #Set solver parameters
        if param['solver'] == 'petsc':
            print('Using Petsc to solve forward model')
            param['solver_param'] = param['solver_petsc']
        elif param['solver'] == 'default':
            print('Using default solver for forward model')
            param['solver_param'] = param['solver_default']
        elif param['solver'] == 'custom':
            print('Using custom solver for forward model')
            param['solver_param'] = param['solver_custom']
        else:
            print('Unrecognized forward solver, using default')
            param['solver_param'] = param['solver_default']

        self.param = param

    def init_surf(self,surf):
        self.surf = project(surf,self.Q)

    def init_bed(self,bed):
        self.bed = project(bed,self.Q)

    def init_thick(self,thick):
        self.thick = project(thick,self.Q)

    def init_alpha(self,alpha):
        self.alpha = project(alpha,self.Q)

    def init_bmelt(self,bmelt):
        self.bmelt = project(bmelt,self.Q)

    def init_vel_obs(self, u, v, mv):
        self.u_obs = project(u,self.Q)
        self.v_obs = project(v,self.Q)
        self.mask_vel = project(mv,self.M)

    def init_mask(self,mask):
        self.mask = project(mask,self.M)

    def gen_thick(self):
        rhoi = self.param['rhoi']
        rhow = self.param['rhow']

        h_diff = self.surf-self.bed
        h_hyd = self.surf*1.0/(1-rhoi/rhow)
        self.thick = project(Min(h_diff,h_hyd),self.Q)

    def gen_ice_mask(self):
        self.mask = project(conditional(gt(self.thick,self.param['tol']),1,0), self.M)

    def gen_vel_mask(self):
        self.mask_vel = project(Constant(1.0), self.M)


    def gen_alpha(self, a_bgd=2000.0, a_lb = 1.0, a_ub = 1.0e4):

        bed = self.bed
        height = self.thick
        g = self.param['g']
        rhoi = self.param['rhoi']
        rhow = self.param['rhow']
        u_obs = self.u_obs
        v_obs = self.v_obs
        U = Max((u_obs**2 + v_obs**2)**(1/2.0),2.0)

        #Flotation Criterion
        height_s = -rhow/rhoi * bed
        fl_ex = conditional(height <= height_s, 1.0, 0.0)

        #Thickness Criterion
        m_d = conditional(height > 0,1.0,0.0)

        #Surface gradient (including where there is no ice surface...)
        R_f = ((1.0 - fl_ex) * bed
               + (fl_ex) * (-rhoi / rhow) * height)
        s = height + R_f
        grads = (s.dx(0)**2.0 + s.dx(1)**2.0)**(1.0/2.0)

        #Calculate alpha, apply bound
        self.alpha_ = ( (1.0 - fl_ex) *rhoi*g*height*grads/U
           + (fl_ex) * a_bgd ) * m_d + (1.0-m_d) * a_bgd


        self.alpha__ = Max(self.alpha_, a_lb)
        self.alpha = Min(self.alpha__, a_ub)
        self.alpha = ln(self.alpha)

        #self.U = interpolate(Expression(('50','50'),degree=1),self.V)


    def gen_domain(self):
        tol = self.param['tol']

        #Mask labels
        self.MASK_ICE = 1   #Ice
        self.MASK_LO = 0    #Land/Ocean
        self.MASK_XD = -10  #Out of domain

        #Cell labels
        self.OMEGA_X     = 0     #exterior to ice sheet
        self.OMEGA_ICE   = 1
        self.OMEGA_ICE_OBS = 2


        #Facet labels
        self.GAMMA_DEF = 0      #default value
        self.GAMMA_LAT = 1      #Value at lateral domain boundaries
        self.GAMMA_TMN = 2      #Value at ice terminus
        self.GAMMA_NF = 3       #No flow dirichlet bc

        #Cell and Facet markers
        self.cf      = CellFunction('size_t',  self.mesh)
        self.ff      = FacetFunction('size_t', self.mesh)

        #Initialize Values
        self.cf.set_all(self.OMEGA_X)
        self.ff.set_all(self.GAMMA_DEF)

        # Build connectivity between facets and cells
        D = self.mesh.topology().dim()
        self.mesh.init(D-1,D)

        #Label ice sheet cells
        for c in cells(self.mesh):
            x_m       = c.midpoint().x()
            y_m       = c.midpoint().y()
            m_xy = self.mask(x_m, y_m)
            mv_xy = self.mask_vel(x_m, y_m)

            #Determine whether cell is in the domain
            if near(m_xy,1, tol) & near(mv_xy,1, tol):
                self.cf[c] = self.OMEGA_ICE_OBS

            elif near(m_xy,1, tol):
                self.cf[c] = self.OMEGA_ICE

        for f in facets(self.mesh):
            x_m      = f.midpoint().x()
            y_m      = f.midpoint().y()
            m_xy = self.mask(x_m, y_m)

            if f.exterior():
                if near(m_xy,1,tol):
                    self.ff[f] = self.GAMMA_LAT

            else:
                #Identify the 2 neighboring cells
                [n1_num,n2_num] = f.entities(D)

                #Mask value of neighbor 1
                n1 = Cell(self.mesh,n1_num)
                n1_x = n1.midpoint().x()
                n1_y = n1.midpoint().y()
                n1_mask = self.mask(n1_x,n1_y)

                #Mask value of neighbor 2
                n2 = Cell(self.mesh,n2_num)
                n2_x = n2.midpoint().x()
                n2_y = n2.midpoint().y()
                n2_mask = self.mask(n2_x,n2_y)

                #Identify facets which to apply terminating boundary condition
                if near(n1_mask + n2_mask,self.MASK_ICE + self.MASK_LO,tol):
                    self.ff[f] = self.GAMMA_TMN


                #Identify facets which to apply no flow boundary condition
                if near(n1_mask+n2_mask,self.MASK_ICE +self.MASK_XD,tol):
                    self.ff[f] = self.GAMMA_NF
