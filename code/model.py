from dolfin import *
import numpy as np
import timeit
from IPython import embed


class model:

    def __init__(self, mesh_in, outdir='./output/'):
        self.mesh = Mesh(mesh_in)
        self.V = VectorFunctionSpace(self.mesh,'Lagrange',2)
        self.Q = FunctionSpace(self.mesh,'Lagrange',2)

        self.U = Function(self.V)
        self.Phi = TestFunction(self.V)
        self.dU = TrialFunction(self.V)

        self.outdir = outdir
        self.init_constants()


    def default_solver_params(self):
        self.solve_param = {'newton_solver' :
                {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'hypre_amg',
                'relative_tolerance'       : 1e-10,
                'relaxation_parameter'     : 1.0,
                'absolute_tolerance'       : 1.0,
                'maximum_iterations'       : 20,
                'error_on_nonconvergence'  : True,
                'krylov_solver'            :
                {
                'monitor_convergence'   : True,
                }}}


    def init_constants(self):

        self.td = 24*60*60.0
        self.ty = 365*24*60*60.0

        self.rhoi =  917.0
        self.rhow =  1000.0

        self.g = 9.81
        self.n = 3.0
        self.eps_rp = 10.0e-5

        self.A = 7.0e-25


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

        self.thick = Function(self.Q)

        h = self.surf.vector()[:] - self.bed.vector()[:]
        h_hyd = self.surf.vector()[:]*1.0/(1-rhoi/rhow)

        self.thick.vector()[:] = np.minimum(h,h_hyd)


    def gen_ice_mask(self):
        tol = 1e-6
        self.mask = Function(self.Q)
        self.mask.vector()[:] = np.abs(self.surf.vector()[:] - self.bed.vector()[:]) > tol

    def gen_boundaries(self):

        #Cell labels
        self.OMEGA_X    = 0     #exterior to ice sheet
        self.OMEGA_GND   = 1   # internal cells over bedrock
        self.OMEGA_FLT   = 2   # internal cells over water

        #Face labels
        self.GAMMA_X = 0    #facets not on boundary
        self.GAMMA_DMN = 1   # domain boundary
        self.GAMMA_GND = 2   # terminus
        self.GAMMA_FLT = 3   # terminus

        #Cell and Facet markers
        self.cf      = CellFunction('size_t',  self.mesh)
        self.ff      = FacetFunction('size_t', self.mesh)

        #Initialize Values
        self.cf.set_all(self.OMEGA_X)
        self.ff.set_all(self.GAMMA_X)

        # Build connectivity between facets and cells
        D = self.mesh.topology().dim()
        self.mesh.init(D-1,D)


        #Label cells
        rhow = self.rhow
        rhoi = self.rhoi

        for c in cells(self.mesh):
            x_m       = c.midpoint().x()
            y_m       = c.midpoint().y()
            mask_xy   = self.mask(x_m, y_m)
            h_xy = self.thick(x_m, y_m)
            bed_xy = self.bed(x_m,y_m)

            #Determine whether the cell is floating, grounded, or not of interest
            if near(mask_xy, 1):
                if h_xy >= rhow*(0-bed_xy)/rhoi:
                    self.cf[c] = self.OMEGA_GND
                else:
                    self.cf[c] = self.OMEGA_FLT


        #Label facets
        for f in facets(self.mesh):
            x_m      = f.midpoint().x()
            y_m      = f.midpoint().y()
            mask_xy = self.mask(x_m, y_m)

            if near(mask_xy,1):

                if f.exterior():
                    self.ff[f] = self.GAMMA_DMN

                else:
                    #Identify the 2 neighboring cells
                    [n1_num,n2_num] = f.entities(D)

                    #Properties of neighbor 1
                    n1 = Cell(self.mesh,n1_num)
                    n1_x = c.midpoint().x()
                    n1_y = c.midpoint().y()
                    n1_mask = self.mask(n1_x,n1_y)
                    n1_bool = near(n1_mask,1)

                    #Properties of neighbor 2
                    n2 = Cell(self.mesh,n2_num)
                    n2_x = c.midpoint().x()
                    n2_y = c.midpoint().y()
                    n2_mask = self.mask(n2_x,n2_y)
                    n2_bool = near(n2_mask,1)

                    #Identify if terminus cell
                    if n1_bool + n2_bool == 1: #XOR
                        #Grounded or Floating
                        bed_xy = self.bed(x_m, y_m)
                        if bed_xy >= 0:
                            self.ff[f] = GAMMA_GND
                        else:
                            self.ff[f] = GAMMA_FLT

        self.set_measures()

    def set_measures(self):

        # create new measures of integration :
        self.dx      = Measure('dx', subdomain_data=self.cf)
        self.ds      = Measure('ds', subdomain_data=self.ff)

        self.dIce_gnd   = self.dx(self.OMEGA_GND)   #grounded
        self.dIce_flt   = self.dx(self.OMEGA_FLT)   #floating
        self.dIce       = self.dIce_gnd + self.dIce_flt #bed

        self.dLat_gnd  = self.ds(self.GAMMA_GND)    #grounded
        self.dLat_flt  = self.ds(self.GAMMA_FLT)    #floating
        self.dLat_dmn  = self.ds(self.GAMMA_DMN)    #dirichlet
