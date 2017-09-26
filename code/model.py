from dolfin import *
import numpy as np
import timeit
from IPython import embed
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
        param['gamma'] =  1          #Cost function scaling parameter

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
                'relaxation_parameter'     : 0.70,
                'absolute_tolerance'       : 10.0,
                'maximum_iterations'       : 50,
                'error_on_nonconvergence'  : False,
                }}

        
        param['inv_options'] = {'disp': True, 'maxiter': 15, 'ftol' : 0.05}

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

    def init_alpha(self,alpha):
        self.alpha = project(alpha,self.Q)

    def init_bmelt(self,bmelt):
        self.bmelt = project(bmelt,self.Q)

    def init_thick(self):
        rhoi = self.param['rhoi']
        rhow = self.param['rhow']

        h_diff = self.surf-self.bed
        h_hyd = self.surf*1.0/(1-rhoi/rhow)
        self.thick = project(Min(h_diff,h_hyd),self.Q)

    def init_vel_obs(self, u, v):
        self.u_obs = project(u,self.Q)
        self.v_obs = project(v,self.Q)

    def gen_ice_mask(self):
        self.mask = project(conditional(gt(self.thick,self.param['tol']),1,0), self.Q)

    def gen_domain(self):

        #Cell labels
        self.OMEGA_X    = 0     #exterior to ice sheet
        self.OMEGA_ICE   = 1    #Ice Sheet

        #Facet labels
        self.GAMMA_DEF = 0      #default value
        self.GAMMA_LAT = 1      #Value at lateral domain boundaries

        #Cell and Facet markers
        self.cf      = CellFunction('size_t',  self.mesh)
        self.ff      = FacetFunction('size_t', self.mesh)

        #Initialize Values
        self.cf.set_all(self.OMEGA_X)
        self.ff.set_all(self.GAMMA_DEF)

        #Label ice sheet cells
        for c in cells(self.mesh):
            x_m       = c.midpoint().x()
            y_m       = c.midpoint().y()
            h_xy = self.thick(x_m, y_m)

            #Determine whether cell is ice covered
            if h_xy > self.param['tol']:
                self.cf[c] = self.OMEGA_ICE

        for f in facets(self.mesh):
            x_m      = f.midpoint().x()
            y_m      = f.midpoint().y()
            h_xy = self.thick(x_m, y_m)

            if f.exterior() and h_xy > self.param['tol']:
                self.ff[f] = self.GAMMA_LAT


        self.set_measures()

    def set_measures(self):
        # create new measures of integration :
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.cf)

        self.dIce   =     self.dx(self.OMEGA_ICE)   #grounded ice
