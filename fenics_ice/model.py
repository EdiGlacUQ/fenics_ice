from fenics import *
from dolfin import *
import ufl
import numpy as np
from pathlib import Path
import scipy.spatial.qhull as qhull
from fenics_ice import inout
from fenics_ice import mesh as fice_mesh
from IPython import embed
from numpy.random import randn
import logging

log = logging.getLogger("fenics_ice")

class model:

    def __init__(self, mesh_in, input_data, param_in):

        #Initiate parameters
        self.params = param_in
        self.input_data = input_data

        #Full mask/mesh
        self.mesh_ext = Mesh(mesh_in)
        M_in = FunctionSpace(self.mesh_ext, 'DG', 0)
        self.mask_ext = self.input_data.interpolate("data_mask", M_in)

        #Generate Domain and Function Spaces
        self.gen_domain()
        self.nm = FacetNormal(self.mesh)
        self.Q = FunctionSpace(self.mesh,'Lagrange',1)

        self.M = FunctionSpace(self.mesh,'DG',0)
        self.RT = FunctionSpace(self.mesh,'RT',1)

        #Based on IsmipC: alpha, beta, and U are periodic.
        if not self.params.mesh.periodic_bc:
            self.Qp = self.Q
            self.V = VectorFunctionSpace(self.mesh, 'Lagrange', 1, dim=2)
        else:
            self.Qp = fice_mesh.get_periodic_space(self.params, self.mesh, dim=1)
            self.V = fice_mesh.get_periodic_space(self.params, self.mesh, dim=2)

        #Default velocity mask and Beta fields
        self.def_vel_mask()
        self.def_B_field()
        self.def_lat_dirichletbc()

    def init_param(self, param_in):
        """
        Sets parameter values to defaults, then overwrites with param_in
        """
        #TODO - UNUSED - DELETE ONCE DEFAULTS EXTRACTED
        #Constants for ice sheet modelling
        param = {}
        param['ty'] = 365*24*60*60.0  #seconds in year
        param['rhoi'] =  917.0      #density ice
        param['rhow'] =   1000.0     #density water
        param['g'] =  9.81           #acceleration due to gravity
        param['n'] =  3.0            #glen's flow law exponent
        param['eps_rp'] =  1e-5      #effective strain regularization
        param['vel_rp'] =  1e-2      #velocity regularization parameter
        param['A'] =  3.5e-25 * param['ty']     #Creep paramater
        param['tol'] =  1e-6         #Tolerance for tests
        param['rc_inv'] =  [0.0]       #regularization constants for inversion

        #Sliding law
        param['sliding_law'] =  0.0  #Alternatively 'weertman'

        #Output
        param['outdir'] = './output/'

        #Timestepping
        param['run_length'] = 1.0
        param['n_steps'] = 24
        param['num_sens'] = 0.0

        #Solver options
        param['picard_params'] = {"nonlinear_solver":"newton",
                    "newton_solver":{"linear_solver":"umfpack",
                    "maximum_iterations":25,
                    "absolute_tolerance":1.0e-8,
                    "relative_tolerance":5.0e-2,
                    "convergence_criterion":"incremental",}}

        param['newton_params'] = {"nonlinear_solver":"newton",
                    "newton_solver":{"linear_solver":"umfpack",
                    "maximum_iterations":25,
                    "absolute_tolerance":1.0e-5,
                    "relative_tolerance":1.0e-5,
                    "convergence_criterion":"incremental",
                    "error_on_nonconvergence":True,}}

        #Boundary Conditions
        param['periodic_bc'] = False

        #Inversion options
        param['inv_options'] = {'disp': True, 'maxiter': 5}

        #Update default values based on input
        param.update(param_in)
        param['dt'] = param['run_length']/param['n_steps']

        self.param = param

    def bglen_to_beta(self,x):
        return sqrt(x)

    def beta_to_bglen(self,x):
        return x*x

    def def_vel_mask(self):
        self.mask_vel_M = project(Constant(0.0), self.M)

    def def_B_field(self):
        """Define beta field from constants in config file"""
        A = self.params.constants.A
        n = self.params.constants.glen_n
        self.beta = project(self.bglen_to_beta(A**(-1.0/n)), self.Qp)
        self.beta_bgd = project(self.bglen_to_beta(A**(-1.0/n)), self.Qp)
        self.beta.rename('beta', 'a Function')

    def def_lat_dirichletbc(self):
        self.latbc = Constant([0.0,0.0])

    def mask_from_data(self):
        self.mask = self.input_data.interpolate("data_mask", self.M)

    def surf_from_data(self):
        self.surf = self.input_data.interpolate("surf", self.Q)

    def bed_from_data(self):
        self.bed = self.input_data.interpolate("bed", self.Q)

    def thick_from_data(self):
        self.H_np = self.input_data.interpolate("thick", self.M)
        self.H_s = self.H_np.copy(deepcopy=True)
        self.H = 0.5*(self.H_np + self.H_s)

    def alpha_from_data(self):
        self.alpha = self.input_data.interpolate("alpha", self.Qp)

    def alpha_from_inversion(self):
        """Get alpha field from inversion step"""
        inversion_file = self.params.io.inversion_file
        outdir = self.params.io.output_dir

        with HDF5File(self.mesh.mpi_comm(),
                      str(Path(outdir)/inversion_file),
                      'r') as infile:
            self.alpha = Function(self.Qp, name='alpha')
            infile.read(self.alpha, 'alpha')

    def beta_from_inversion(self):
        """Get beta field from inversion step"""
        inversion_file = self.params.io.inversion_file
        outdir = self.params.io.output_dir

        with HDF5File(self.mesh.mpi_comm(),
                      str(Path(outdir)/inversion_file),
                      'r') as infile:
            self.beta = Function(self.Qp, name='beta')
            infile.read(self.beta, 'beta')
            self.beta_bgd = self.beta.copy(deepcopy=True)

    def bmelt_from_data(self):
        self.bmelt = self.input_data.interpolate("bmelt", self.M, 0)

    def smb_from_data(self):
        self.smb = self.input_data.interpolate("smb", self.M, 0)

    def bglen_from_data(self):
        self.bglen = self.input_data.interpolate("bglen", self.Q, 0)

    def init_beta(self, beta, pert=False):
        self.beta_bgd = project(beta,self.Qp)
        self.beta = project(beta,self.Qp)
        if pert:
            #Perturbed field for nonzero gradient at first step of inversion
            bv = self.beta.vector().get_local()
            pert_vec = 0.001*bv*randn(bv.size)
            self.beta.vector().set_local(bv + pert_vec)
            self.beta.vector().apply('insert')

        self.beta.rename('beta', 'a Function')

    def init_vel_obs(self):
        """
        Read velocity observations & uncertainty from HDF5 file

        Additionally interpolates these arbitrarily spaced data
        onto self.Q for use as boundary conditions etc
        """

        # Read the obs from HDF5 file
        # Generates self.u_obs, self.v_obs, self.u_std, self.v_std,
        # self.uv_obs_pts, self.mask_vel
        inout.read_vel_obs(self.params, self)

        # Functions for repeated ungridded interpolation
        def interp_weights(xy, uv, d=2):
            """Compute the nearest vertices & weights (for reuse)"""
            tri = qhull.Delaunay(xy)
            simplex = tri.find_simplex(uv)
            vertices = np.take(tri.simplices, simplex, axis=0)
            temp = np.take(tri.transform, simplex, axis=0)
            delta = uv - temp[:, d]
            bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
            return vertices, np.hstack((bary, 1 - bary.sum(axis=1,
                                                           keepdims=True)))

        def interpolate(values, vtx, wts):
            """Bilinear interpolation, given vertices & weights above"""
            return np.einsum('nj,nj->n', np.take(values, vtx), wts)

        # Grab coordinates of both Lagrangian & DG function spaces
        # and compute (once) the interpolating arrays
        Q_coords = self.Q.tabulate_dof_coordinates()
        M_coords = self.M.tabulate_dof_coordinates()
        vtx_Q, wts_Q = interp_weights(self.uv_obs_pts, Q_coords)
        vtx_M, wts_M = interp_weights(self.uv_obs_pts, M_coords)

        # Define new functions to hold results
        self.u_obs_Q = Function(self.Q)
        self.v_obs_Q = Function(self.Q)
        self.u_std_Q = Function(self.Q)
        self.v_std_Q = Function(self.Q)
        # self.mask_vel_Q = Function(self.Q)

        self.u_obs_M = Function(self.M)
        self.v_obs_M = Function(self.M)
        # self.u_std_M = Function(self.M)
        # self.v_std_M = Function(self.M)
        self.mask_vel_M = Function(self.M)

        # Fill via interpolation
        self.u_obs_Q.vector()[:] = interpolate(self.u_obs, vtx_Q, wts_Q)
        self.v_obs_Q.vector()[:] = interpolate(self.v_obs, vtx_Q, wts_Q)
        self.u_std_Q.vector()[:] = interpolate(self.u_std, vtx_Q, wts_Q)
        self.v_std_Q.vector()[:] = interpolate(self.v_std, vtx_Q, wts_Q)
        # self.mask_vel_Q.vector()[:] = interpolate(self.mask_vel, vtx_Q, wts_Q)

        self.u_obs_M.vector()[:] = interpolate(self.u_obs, vtx_M, wts_M)
        self.v_obs_M.vector()[:] = interpolate(self.v_obs, vtx_M, wts_M)
        # self.u_std_M.vector()[:] = interpolate(self.u_std, vtx_M, wts_M)
        # self.v_std_M.vector()[:] = interpolate(self.v_std, vtx_M, wts_M)
        self.mask_vel_M.vector()[:] = interpolate(self.mask_vel, vtx_M, wts_M)

    def init_vel_obs_old(self, u, v, mv, ustd=Constant(1.0),
                         vstd=Constant(1.0), ls = False):
        """
        Set up velocity observations for inversion

        Approach here involves velocity defined on functions (on mesh) which
        are projected onto the current model mesh. uv_obs_pts is a set of numpy
        coordinates which can be arbitrarily defined, and obs are then
        interpolated (again) onto these points in comp_J_inv.
        """
        self.u_obs = project(u,self.M)
        self.v_obs = project(v,self.M)
        self.mask_vel = project(mv,self.M)
        self.u_std = project(ustd,self.M)
        self.v_std = project(vstd,self.M)

        if ls:
            mc = self.mesh.coordinates()
            xmin = mc[:,0].min()
            xmax = mc[:,0].max()

            ymin = mc[:,1].min()
            ymax = mc[:,1].max()

            xc = np.arange(xmin + ls/2.0, xmax, ls) 
            yc = np.arange(ymin + ls/2.0, ymax, ls)

            self.uv_obs_pts = np.transpose([np.tile(xc, len(yc)), np.repeat(yc, len(xc))])

        else:
            self.uv_obs_pts = self.M.tabulate_dof_coordinates().reshape(-1,2)

        

    def init_lat_dirichletbc(self):
        """
        Set lateral vel BC from obs
        """

        latbc = Function(self.V)
        assign(latbc.sub(0), self.u_obs_Q)
        assign(latbc.sub(1), self.v_obs_Q)

        self.latbc = latbc

    def gen_thick(self):
        rhoi = self.params.constants.rhoi
        rhow = self.params.constants.rhow

        h_diff = self.surf-self.bed
        h_hyd = self.surf*1.0/(1-rhoi/rhow)
        self.H = project(Min(h_diff,h_hyd),self.M)

    def gen_surf(self):
        rhoi = self.params.constants.rhoi
        rhow = self.params.constants.rhow
        bed = self.bed
        H = self.H

        H_s = -rhow/rhoi * bed
        fl_ex = conditional(H <= H_s, 1.0, 0.0)

        self.surf = project((1-fl_ex)*(bed+H) + (fl_ex)*H*(1-rhoi/rhow), self.Q)

    def gen_ice_mask(self):
        """
        UNUSED - would overwrite self.mask w/ extent of ice sheet (H > 0)
        """
        tol = self.params.constants.float_eps
        self.mask = project(conditional(gt(self.H,tol),1,0), self.M)

    def gen_alpha(self, a_bgd=500.0, a_lb = 1e2, a_ub = 1e4):
        """
        Initial guess for alpha (slip coeff)
        """

        bed = self.bed
        H = self.H
        g = self.params.constants.g
        rhoi = self.params.constants.rhoi
        rhow = self.params.constants.rhow
        u_obs = self.u_obs_M
        v_obs = self.v_obs_M
        vel_rp = self.params.constants.vel_rp

        U = ufl.Max((u_obs**2 + v_obs**2)**(1/2.0), 50.0)

        #Flotation Criterion
        H_s = -rhow/rhoi * bed
        fl_ex = conditional(H <= H_s, 1.0, 0.0)

        #Thickness Criterion
        m_d = conditional(H > 0,1.0,0.0)

        #Calculate surface gradient
        R_f = ((1.0 - fl_ex) * bed
               + (fl_ex) * (-rhoi / rhow) * H)

        s_ = ufl.Max(H + R_f,0)
        s = project(s_,self.Q)
        grads = (s.dx(0)**2.0 + s.dx(1)**2.0)**(1.0/2.0)

        #Calculate alpha, apply background, apply bound
        B2_ = ( (1.0 - fl_ex) *rhoi*g*H*grads/U
           + (fl_ex) * a_bgd ) * m_d + (1.0-m_d) * a_bgd


        B2_tmp1 = ufl.Max(B2_, a_lb)
        B2_tmp2 = ufl.Min(B2_tmp1, a_ub)

        sl = self.params.ice_dynamics.sliding_law
        if sl == 'linear':
            alpha = sqrt(B2_tmp2)
        elif sl == 'weertman':
            N = (1-fl_ex)*(H*rhoi*g + ufl.Min(bed,0.0)*rhow*g)
            U_mag = sqrt(u_obs**2 + v_obs**2 + vel_rp**2)
            alpha = (1-fl_ex)*sqrt(B2_tmp2 * ufl.Max(N, 0.01)**(-1.0/3.0) * U_mag**(2.0/3.0))

        self.alpha = project(alpha,self.Qp)
        self.alpha.rename('alpha', 'a Function')


    def gen_domain(self):
        """
        Takes the input mesh (self.mesh_ext) and produces the submesh
        where mask==1, which becomes self.mesh
        """
        tol = self.params.constants.float_eps
        cf_mask = MeshFunction('size_t',  self.mesh_ext, self.mesh_ext.geometric_dimension())

        for c in cells(self.mesh_ext):
            x_m       = c.midpoint().x()
            y_m       = c.midpoint().y()
            m_xy = self.mask_ext(x_m, y_m)

            #Determine whether cell is in the domain
            if near(m_xy,1, tol):
                cf_mask[c] = 1

        self.mesh = SubMesh(self.mesh_ext, cf_mask, 1)


    def label_domain(self):
        tol = self.params.constants.float_eps
        bed = self.bed
        H = self.H
        g = self.params.constants.g
        rhoi = self.params.constants.rhoi
        rhow = self.params.constants.rhow

        #Flotation Criterion
        H_s = -rhow/rhoi * bed
        fl_ex_ = conditional(H <= H_s, 1.0, 0.0)
        fl_ex = project(fl_ex_, self.M)

        #Mask labels
        self.MASK_ICE           = 1 #Ice
        self.MASK_LO            = 0 #Land/Ocean
        self.MASK_XD            = -10 #Out of domain

        #Cell labels
        self.OMEGA_DEF          = 0     #default value; should not appear in cc after processing
        self.OMEGA_ICE_FLT      = 1
        self.OMEGA_ICE_GND      = 2
        self.OMEGA_ICE_FLT_OBS  = 3
        self.OMEGA_ICE_GND_OBS  = 4

        #Facet labels
        self.GAMMA_DEF          = 0 #default value, appears in interior cells
        self.GAMMA_LAT          = 1 #Value at lateral domain boundaries
        self.GAMMA_TMN          = 2 #Value at ice terminus
        self.GAMMA_NF           = 3 #No flow dirichlet bc




        #Cell and Facet Markers
        self.cf      = MeshFunction('size_t',  self.mesh, self.mesh.geometric_dimension())
        self.ff      = MeshFunction('size_t', self.mesh, self.mesh.geometric_dimension() - 1)


        #Initialize Values
        self.cf.set_all(self.OMEGA_DEF)
        self.ff.set_all(self.GAMMA_DEF)


        # Build connectivity between facets and cells
        D = self.mesh.topology().dim()
        self.mesh.init(D-1,D)

        #Label ice sheet cells
        for c in cells(self.mesh):
            x_m       = c.midpoint().x()
            y_m       = c.midpoint().y()
            m_xy = self.mask_ext(x_m, y_m)
            mv_xy = self.mask_vel_M(x_m, y_m)
            fl_xy = fl_ex(x_m, y_m)

            #Determine whether cell is in the domain
            if near(m_xy,1.0, tol) & near(mv_xy,1.0, tol):
                if fl_xy:
                    self.cf[c] = self.OMEGA_ICE_FLT_OBS
                else:
                    self.cf[c] = self.OMEGA_ICE_GND_OBS
            elif near(m_xy,1.0, tol):
                if fl_xy:
                    self.cf[c] = self.OMEGA_ICE_FLT
                else:
                    self.cf[c] = self.OMEGA_ICE_GND

        for f in facets(self.mesh):

            #Facet facet label based on mask and corresponding bc
            if f.exterior():
                mask        = self.mask_ext
                x_m         = f.midpoint().x()
                y_m         = f.midpoint().y()
                tol         = 1e-2

                CC = mask(x_m,y_m)

                try:
                    CE = mask(x_m + tol,y_m)
                except:
                    CE = np.Inf
                try:
                    CW = mask(x_m - tol ,y_m)
                except:
                    CW = np.Inf
                try:
                    CN = mask(x_m, y_m + tol)
                except:
                    CN = np.Inf
                try:
                    CS = mask(x_m, y_m - tol)
                except:
                    CS = np.Inf

                mv = np.min([CC, CE, CW, CN, CS])

                if near(mv,self.MASK_ICE,tol):
                    self.ff[f] = self.GAMMA_LAT

                elif near(mv,self.MASK_LO,tol):
                    self.ff[f] = self.GAMMA_TMN

                elif near(mv,self.MASK_XD,tol):
                    self.ff[f] = self.GAMMA_NF



class PeriodicBoundary(SubDomain):
    def __init__(self,L):
        self.L = L
        super(PeriodicBoundary, self).__init__()

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, L) and (L, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                (not ((near(x[0], 0) and near(x[1], self.L)) or
                        (near(x[0], self.L) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.L) and near(x[1], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1] - self.L
        elif near(x[0], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1]
        else:   # near(x[1], L)
            y[0] = x[0]
            y[1] = x[1] - self.L
