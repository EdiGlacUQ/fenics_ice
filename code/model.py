from dolfin import *
import numpy as np
import timeit
from IPython import embed
import matplotlib.pyplot as plt


class model:

    def __init__(self, mesh_in, outdir='./output/',eq_def=1):
        self.mesh = Mesh(mesh_in)
        self.nm = FacetNormal(self.mesh)
        self.V = VectorFunctionSpace(self.mesh,'Lagrange',1,dim=2)
        self.Q = FunctionSpace(self.mesh,'Lagrange',1)
        self.M = FunctionSpace(self.mesh,'DG',0)

        self.U = Function(self.V)
        self.dU = TrialFunction(self.V)
        self.Phi = TestFunction(self.V)

        self.outdir = outdir
        self.eq_def = eq_def

        self.init_constants()


    def default_solver_params(self):

        #Use PETSC. Advantage is the backtracking line search
        PETScOptions().set("snes_linesearch_alpha",1e-6)
        self.solve_param1 = {'nonlinear_solver'      : 'snes',
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
        self.solve_param = {'newton_solver' :
                {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'jacobi',
                'relative_tolerance'       : 1e-15,
                'relaxation_parameter'     : 0.70,
                'absolute_tolerance'       : 10.0,
                'maximum_iterations'       : 50,
                'error_on_nonconvergence'  : False,
                }}


    def init_constants(self):
        self.ty = 31556926.0

        self.rhoi =  917.0
        self.rhow =  1000.0
        self.delta = 1.0 - self.rhoi/self.rhow

        self.g = 9.81
        self.n = 3.0
        self.eps_rp = 1e-5

        self.A = 10**(-16)

        self.tol = 1e-6

    def init_surf(self,surf):
        self.surf = project(surf,self.Q)

    def init_bed(self,bed):
        self.bed = project(bed,self.Q)

    def init_bdrag(self,bdrag):
        self.bdrag = project(bdrag,self.Q)

    def init_bmelt(self,bmelt):
        self.bmelt = project(bmelt,self.Q)

    def init_thick(self):
        rhoi = self.rhoi
        rhow = self.rhow

        h_diff = self.surf-self.bed
        h_hyd = self.surf*1.0/(1-rhoi/rhow)
        self.thick = project(Min(h_diff,h_hyd),self.Q)

    def gen_ice_mask(self):
        self.mask = project(conditional(gt(self.thick,self.tol),1,0), self.Q)

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
            if h_xy > self.tol:
                self.cf[c] = self.OMEGA_ICE

        for f in facets(self.mesh):
            x_m      = f.midpoint().x()
            y_m      = f.midpoint().y()
            h_xy = self.thick(x_m, y_m)

            if f.exterior() and h_xy > self.tol:
                self.ff[f] = self.GAMMA_LAT


        self.set_measures()

    def set_measures(self):
        # create new measures of integration :
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.cf)

        self.dIce   =     self.dx(self.OMEGA_ICE)   #grounded ice
