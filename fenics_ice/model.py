# For fenics_ice copyright information see ACKNOWLEDGEMENTS in the fenics_ice
# root directory

# This file is part of fenics_ice.
#
# fenics_ice is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# fenics_ice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from .backend import *

from . import inout, prior
from . import mesh as fice_mesh

import os.path
import ufl
import numpy as np
from pathlib import Path
from numpy.random import randn
import logging

log = logging.getLogger("fenics_ice")

        # Functions for repeated ungridded interpolation
        # TODO - this will not handle extrapolation/missing data
        # nicely - unfound simplex are returned '-1' which takes the last
        # tri.simplices...

        # at the moment i have moved these from vel_obs_from_data, so they 
        # can be called directly from a run script.
        # the ismipc test, which calls this function, still seems to perform fine
        # but this refactoring may make things less efficient.
def interp_weights(xy, uv, periodic_bc, d=2):
            """Compute the nearest vertices & weights (for reuse)"""
            from scipy.spatial import Delaunay
            tri = Delaunay(xy)
            simplex = tri.find_simplex(uv)

            if not np.all(simplex >= 0):
                if not periodic_bc:
                    log.warning("Some points missing in interpolation "
                              "of velocity obs to function space.")
                else:
                    log.warning("Some points missing in interpolation "
                                "of velocity obs to function space.")

            vertices = np.take(tri.simplices, simplex, axis=0)
            temp = np.take(tri.transform, simplex, axis=0)
            delta = uv - temp[:, d]
            bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
            return vertices, np.hstack((bary, 1 - bary.sum(axis=1,
                                                           keepdims=True)))

def interpolate(values, vtx, wts):
            """Bilinear interpolation, given vertices & weights above"""
            return np.einsum('nj,nj->n', np.take(values, vtx), wts)

class model:
    """
    The 'model' object is the core of any fenics_ice simulation. It handles loading input
    data into fields, generating FunctionSpaces & Functions, marking boundary conditions.

    It has a self.solvers list which, at present, will only ever point to a single solver
    object, defined in solver.py
    """
    def __init__(self, mesh_in, input_data, param_in, init_fields=True,
                 init_vel_obs=True):

        # Initiate parameters
        self.vel_obs = None
        self.params = param_in
        self.input_data = input_data
        self.solvers = []
        self.parallel = mesh_in.mpi_comm().size > 1

        # Generate Domain and Function Spaces
        self.mesh = mesh_in
        self.nm = FacetNormal(self.mesh)
        self.Q = FunctionSpace(self.mesh, 'Lagrange', 1)

        self.M = FunctionSpace(self.mesh, 'DG', 0)
        # below definition is for a dedicated DG(0) 
        # thickness and bed. We should allow flexibility 
        # for self.M to be CG
        self.M2 = FunctionSpace(self.mesh, 'DG', 0)
        self.RT = FunctionSpace(self.mesh, 'RT', 1)

        # Based on IsmipC: alpha, beta, and U are periodic.
        if not self.params.mesh.periodic_bc:
            self.Qp = self.Q
            self.V = VectorFunctionSpace(self.mesh, 'Lagrange', 1, dim=2)
        else:
            self.Qp = fice_mesh.get_periodic_space(self.params, self.mesh, dim=1)
            self.V = fice_mesh.get_periodic_space(self.params, self.mesh, dim=2)

        # Define control functions
        # Note - these are potentially distinct from solver.alpha, .beta
        # because solver may use either two CG spaces or a mixed CGxCG space
        self.alpha = Function(self.Qp, name='alpha')
        self.beta = Function(self.Qp, name='beta')
        self.beta_bgd = Function(self.Qp, name='beta_bgd')

        # Default velocity mask and Beta fields
        self.def_vel_mask()
        self.def_B_field()
        self.def_lat_dirichletbc()

        self.mark_BCs()
        if init_fields:
            self.init_fields_from_data()
        else:  # initialize these to avoid complaint on solver creation
            self.bed = None
            self.bed_DG = None
            self.bmelt = None
            self.smb = None
            self.H = None
            self.H_DG = None
            self.H_np = None
            self.surf = None
            self.Umag_DG = None

        if init_vel_obs:
            # Load the velocity observations
            self.vel_obs_from_data()
            # Overwrite Constant(0,0) from def_lat_dirichletbc w/ obs
            self.init_lat_dirichletbc()

        self.Q_sigma = None
        self.Q_sigma_prior = None
        self.t_sens = None
        self.cntrl_sigma = None
        self.cntrl_sigma_prior = None

    @staticmethod
    def bglen_to_beta(x):
        return sqrt(x)

    @staticmethod
    def beta_to_bglen(x):
        return x*x

    def init_fields_from_data(self):
        """Create functions for input data (geom, smb, etc)"""

        min_thick = self.params.ice_dynamics.min_thickness

        self.bed = self.field_from_data("bed", self.Q)
        self.bmelt = self.field_from_data("bmelt", self.M, default=0.0)
        self.smb = self.field_from_data("smb", self.M, default=0.0)
        if (self.params.mass_solve.use_cg_thickness and self.params.mesh.periodic_bc):
         self.H_np = self.field_from_data("thick", self.Qp, min_val=min_thick)
        else:
         self.H_np = self.field_from_data("thick", self.M, min_val=min_thick)

        if self.params.melt.use_melt_parameterisation:
       
         melt_depth_therm_const: float = -999.0
         melt_max_const: float = -999.0

         if (self.params.melt.melt_depth_therm_const == -999.0 or \
          self.params.melt.melt_max_const == -999.0):
          melt_depth = 1.e6
          melt_max = 0.
         else:
          melt_depth = self.params.melt.melt_depth_therm_const
          melt_max = self.params.melt.melt_max_const
        
         self.melt_depth_therm = self.field_from_data("melt_depth_therm", self.M2, default=melt_depth, \
          method='nearest')
         self.melt_max = self.field_from_data("melt_max", self.M2, default=melt_max, method='nearest')

        self.H = self.H_np.copy(deepcopy=True)
        self.H.rename("thick_H", "")
        self.H_DG = Function(self.M2, name="H_DG")
        self.bed_DG = Function(self.M2, name="bed_DG")
        self.H_DG.assign(project(self.H,self.M2))        
        self.bed_DG.assign(project(self.bed,self.M2))

        self.gen_surf()  # surf = bed + thick

    def def_vel_mask(self):
        self.mask_vel_M = project(Constant(0.0), self.M)

    def def_B_field(self):
        """Define beta field from constants in config file"""
        A = self.params.constants.A
        n = self.params.constants.glen_n

        beta = project(self.bglen_to_beta(A**(-1.0/n)), self.Qp)

        self.beta.assign(beta)
        function_update_state(self.beta)
        self.beta_bgd.assign(beta)
        function_update_state(self.beta_bgd)

    def def_lat_dirichletbc(self):
        """Homogenous dirichlet conditions on lateral boundaries"""
        self.latbc = Constant([0.0, 0.0])

    def field_from_data(self, name, space, **kwargs):
        """Interpolate a named field from input data"""
        return self.input_data.interpolate(name, space, **kwargs)

    def alpha_from_data(self):
        """Get alpha field from initial input data (run_momsolve only)"""
        self.alpha.assign(self.input_data.interpolate("alpha", self.Qp))
        function_update_state(self.alpha)

    def bglen_from_data(self, mask_only=False):
        """Get bglen field from initial input data"""

        if not mask_only:
          self.bglen = self.input_data.interpolate("Bglen", self.Q)

        """Get bglen mask field from input data"""
        bglen_mask_CG = self.input_data.interpolate("Bglenmask", self.Q, default=1.0)
        self.bglen_mask = Function(self.M, name="bglen_mask")

        dofmap_M = self.M.dofmap()
        dofmap_Q = self.Q.dofmap()

        """bglen_mask is currently 1 where there is data, zero elsewhere"""
        temp_vec = bglen_mask_CG.vector().gather(np.arange(self.Q.dim()))
        bmask_loc = np.ones(self.bglen_mask.vector().local_size(), dtype=np.float64)
        for cell in range(self.mesh.num_cells()):
          if not Cell(self.mesh, cell).is_ghost():
            local_dofs = dofmap_Q.cell_dofs(cell)
            global_dofs = np.full(np.shape(local_dofs),-1,dtype=int)
            for dof in range(len(local_dofs)):
              global_dofs[dof] = dofmap_Q.local_to_global_index(local_dofs[dof])
            if (temp_vec[global_dofs] < 1.0-1.0e-10).any():
              bmask_loc[dofmap_M.cell_dofs(cell)] = 0

        self.bglen_mask.vector().set_local(bmask_loc)
        """Following appears in inout.interpolate"""
        self.bglen_mask.vector().apply("insert")

    def alpha_from_inversion(self):
        """Get alpha field from inversion step"""
        inversion_file = self.params.io.inversion_file

        phase_suffix = self.params.inversion.phase_suffix
        if len(phase_suffix) > 0:
            inversion_file = self.params.io.run_name + phase_suffix + '_invout.h5'

        outdir = Path(self.params.io.output_dir) / \
                 self.params.inversion.phase_name / \
                 self.params.inversion.phase_suffix

        with HDF5File(self.mesh.mpi_comm(),
                      str(outdir/inversion_file),
                      'r') as infile:
            infile.read(self.alpha, 'alpha')
            function_update_state(self.alpha)

    def beta_from_inversion(self):
        """Get beta field from inversion step"""
        inversion_file = self.params.io.inversion_file

        phase_suffix = self.params.inversion.phase_suffix
        if len(phase_suffix) > 0:
            inversion_file = self.params.io.run_name + phase_suffix + '_invout.h5'

        outdir = Path(self.params.io.output_dir) / \
                 self.params.inversion.phase_name / \
                 self.params.inversion.phase_suffix

        with HDF5File(self.mesh.mpi_comm(),
                      str(outdir/inversion_file),
                      'r') as infile:
            infile.read(self.beta, 'beta')
            function_update_state(self.beta)

        self.beta_bgd.assign(self.beta)
        function_update_state(self.beta_bgd)

    def init_beta(self, beta, pert=False):
        """
        Define the beta field from input

        Optionally perturb the field slightly to prevent zero gradient
        on first step of beta inversion.
        """

        beta_proj = project(beta, self.Qp)
        self.beta.assign(beta_proj)
        function_update_state(self.beta)
        self.beta_bgd.assign(beta_proj)
        function_update_state(self.beta_bgd)

        write_diag = self.params.io.write_diagnostics
        if write_diag:
            diag_dir = self.params.io.diagnostics_dir
            phase_suffix = self.params.inversion.phase_suffix
            phase_name = self.params.inversion.phase_name
            inout.write_variable(self.beta,
                                 self.params,
                                 name="beta_init_guess",
                                 outdir=diag_dir,
                                 phase_name=phase_name,
                                 phase_suffix=phase_suffix)

        # TODO - tidy this up properly (remove pert arg)
        # if pert:
        #     # Perturbed field for nonzero gradient at first step of inversion
        #     bv = self.beta.vector().get_local()
        #     pert_vec = 0.001*bv*randn(bv.size)
        #     self.beta.vector().set_local(bv + pert_vec)
        #     self.beta.vector().apply('insert')
        #     function_update_state(self.beta)


    def vel_obs_from_data(self):
        """
        Reads ice velocity observations from HDF5 file.
        - Additionally interpolates composite velocities
            (gridded spaced data) to mesh coordinates
            to be use in setting the boundary conditions
            and alpha initialisation.
        - Velocities in a cloud point format are kept to compute
            the value of the cost function only if
            params.inversion.use_cloud_point_velocities = true
            else we use composite velocities

        Expects an HDF5 file with the following list of variables
            u_obs, v_obs, u_std, v_std, mask_vel, x, y (as default)
            and if params.inversion.use_cloud_point_velocities = true
        expects the variables above plus cloud point data with the name:
            u_cloud, v_cloud, u_cloud_std, v_cloud_std, x_could, y_could
        Everything without nan's and in a tuple format e.g. x -> (values, )
        """
        infile = Path(self.params.io.input_dir) / self.params.obs.vel_file
        if self.params.inversion.use_cloud_point_velocities:
            inout.read_vel_obs(infile,
                               model=self,
                               use_cloud_point=self.params.inversion.use_cloud_point_velocities)
        else:
            inout.read_vel_obs(infile, model=self)

        # Grab coordinates of both Lagrangian & DG function spaces
        # and compute (once) the interpolating arrays
        Q_coords = self.Q.tabulate_dof_coordinates()
        M_coords = self.M.tabulate_dof_coordinates()

        vtx_Q, wts_Q = interp_weights(self.vel_obs['uv_comp_pts'],
                                      Q_coords, self.params.mesh.periodic_bc)
        vtx_M, wts_M = interp_weights(self.vel_obs['uv_comp_pts'],
                                      M_coords, self.params.mesh.periodic_bc)

        # Define new functions to hold results
        self.u_obs_Q = Function(self.Q, name="u_obs")
        self.v_obs_Q = Function(self.Q, name="v_obs")
        self.u_std_Q = Function(self.Q, name="u_std")
        self.v_std_Q = Function(self.Q, name="v_std")
        # self.mask_vel_Q = Function(self.Q)

        self.u_obs_M = Function(self.M, name="u_obs")
        self.v_obs_M = Function(self.M, name="v_obs")
        # self.u_std_M = Function(self.M)
        # self.v_std_M = Function(self.M)
        self.mask_vel_M = Function(self.M, name="mask_vel")

        # Fill via interpolation
        self.u_obs_Q.vector()[:] = interpolate(self.vel_obs['u_comp'], vtx_Q, wts_Q)
        self.v_obs_Q.vector()[:] = interpolate(self.vel_obs['v_comp'], vtx_Q, wts_Q)
        self.u_std_Q.vector()[:] = interpolate(self.vel_obs['u_comp_std'], vtx_Q, wts_Q)
        self.v_std_Q.vector()[:] = interpolate(self.vel_obs['v_comp_std'], vtx_Q, wts_Q)
        # self.mask_vel_Q.vector()[:] = interpolate(self.mask_vel, vtx_Q, wts_Q)

        self.u_obs_M.vector()[:] = interpolate(self.vel_obs['u_comp'], vtx_M, wts_M)
        self.v_obs_M.vector()[:] = interpolate(self.vel_obs['v_comp'], vtx_M, wts_M)
        # self.u_std_M.vector()[:] = interpolate(self.u_std, vtx_M, wts_M)
        # self.v_std_M.vector()[:] = interpolate(self.v_std, vtx_M, wts_M)
        # IMPORTANT! this mask is not the vel mask of cloud point observations
        # it is the mask from the composite velocities
        self.mask_vel_M.vector()[:] = interpolate(self.vel_obs['mask_vel'], vtx_M, wts_M)

        # We need to do the same as above but for cloud point data
        # so we can write out a nicer output in the mesh coordinates
        vtx_Q_c, wts_Q_c = interp_weights(self.vel_obs['uv_obs_pts'],
                                      Q_coords, self.params.mesh.periodic_bc)

        # Define new functions to hold results
        self.u_cloud_Q = Function(self.Q, name="u_obs_cloud")
        self.v_cloud_Q = Function(self.Q, name="v_obs_cloud")
        self.u_std_cloud_Q = Function(self.Q, name="u_std_cloud")
        self.v_std_cloud_Q = Function(self.Q, name="v_std_cloud")

        # Fill via interpolation
        self.u_cloud_Q.vector()[:] = interpolate(self.vel_obs['u_obs'], vtx_Q_c, wts_Q_c)
        self.v_cloud_Q.vector()[:] = interpolate(self.vel_obs['v_obs'], vtx_Q_c, wts_Q_c)
        self.u_std_cloud_Q.vector()[:] = interpolate(self.vel_obs['u_std'], vtx_Q_c, wts_Q_c)
        self.v_std_cloud_Q.vector()[:] = interpolate(self.vel_obs['v_std'], vtx_Q_c, wts_Q_c)


    def init_vel_obs_old(self, u, v, mv, ustd=Constant(1.0),
                         vstd=Constant(1.0), ls=False):
        """
        Set up velocity observations for inversion

        Approach here involves velocity defined on functions (on mesh) which
        are projected onto the current model mesh. uv_obs_pts is a set of numpy
        coordinates which can be arbitrarily defined, and obs are then
        interpolated (again) onto these points in comp_J_inv.
        """
        self.u_obs = project(u, self.M)
        self.v_obs = project(v, self.M)
        self.mask_vel = project(mv, self.M)
        self.u_std = project(ustd, self.M)
        self.v_std = project(vstd, self.M)

        if ls:
            mc = self.mesh.coordinates()
            xmin = mc[:, 0].min()
            xmax = mc[:, 0].max()

            ymin = mc[:, 1].min()
            ymax = mc[:, 1].max()

            xc = np.arange(xmin + ls/2.0, xmax, ls)
            yc = np.arange(ymin + ls/2.0, ymax, ls)

            self.uv_obs_pts = np.transpose([np.tile(xc, len(yc)), np.repeat(yc, len(xc))])

        else:
            self.uv_obs_pts = self.M.tabulate_dof_coordinates().reshape(-1, 2)

    def init_lat_dirichletbc(self):
        """Set lateral vel BC from obs"""
        latbc = Function(self.V)
        assign(latbc.sub(0), self.u_obs_Q)
        assign(latbc.sub(1), self.v_obs_Q)

        self.latbc = latbc

    def gen_thick(self):
        rhoi = self.params.constants.rhoi
        rhow = self.params.constants.rhow

        h_diff = self.surf-self.bed
        h_hyd = self.surf*1.0/(1-rhoi/rhow)
        self.H = project(Min(h_diff, h_hyd), self.M)

    def gen_surf(self):
        rhoi = self.params.constants.rhoi
        rhow = self.params.constants.rhow
        bed = self.bed
        H = self.H

        H_flt = -rhow/rhoi * bed
        fl_ex = conditional(H <= H_flt, 1.0, 0.0)

        self.surf = project((1-fl_ex)*(bed+H) + (fl_ex)*H*(1-rhoi/rhow), self.Q)
        self.surf.rename("surf", "")

    def bdrag_to_alpha(self, B2):
        """Convert basal drag to alpha"""
        sl = self.params.ice_dynamics.sliding_law


        if sl == 'linear':
            alpha = sqrt(B2)

        elif sl  in ['budd','corn']:
            bed = self.bed
            H = self.H
            g = self.params.constants.g
            rhoi = self.params.constants.rhoi
            rhow = self.params.constants.rhow
            H_flt = -rhow/rhoi * bed
            u_obs = self.u_obs_M
            v_obs = self.v_obs_M
            vel_rp = self.params.constants.vel_rp

            # Flotation Criterion
            fl_ex = conditional(H <= H_flt, 1.0, 0.0)
            N = (1-fl_ex)*(H*rhoi*g + ufl.Min(bed, 0.0)*rhow*g)
            U_mag = sqrt(u_obs**2 + v_obs**2 + vel_rp**2)
            
            if sl == 'budd':
                alpha = (1-fl_ex)*sqrt(B2 * ufl.Max(N, 0.01)**(-1.0/3.0) * U_mag**(2.0/3.0))

            elif sl == 'corn':

                # the relationship between alpha and B2 is too nontrivial to "invert", and an exact
                # solution is not sought. Rather, since the sliding law is expected to deviate from 
                # the weertman law (B2 = alpha^2 U^(-2/3)) only within a few km of the grounding line,
                # we initialise based on the weertman sliding law
                alpha = (1-fl_ex)*sqrt(B2 * U_mag**(2.0/3.0))

        else:
            raise RuntimeError(f"Invalid sliding law: '{sl:s}'")

        return alpha

    def gen_alpha(self, a_bgd=500.0, a_lb=1e2, a_ub=1e4):
        """Generate initial guess for alpha (slip coeff)"""

        method = self.params.inversion.initial_guess_alpha_method.lower()
        if method == "wearing":
            # Martin Wearing's: alpha = 358.4 - 26.9*log(17.9*speed_obs);
            u_obs = self.u_obs_M
            v_obs = self.v_obs_M
            vel_rp = self.params.constants.vel_rp
            U_mag = sqrt(u_obs**2 + v_obs**2 + vel_rp**2)
            # U_mag = ufl.Max((u_obs**2 + v_obs**2)**(1/2.0), 50.0)

            B2 = 358.4 - 26.9*ln(17.9*U_mag)
            alpha = self.bdrag_to_alpha(B2)
            self.alpha.assign(project(alpha, self.Qp))

        elif method == "sia":

            bed = self.bed
            H = self.H
            g = self.params.constants.g
            rhoi = self.params.constants.rhoi
            rhow = self.params.constants.rhow
            u_obs = self.u_obs_M
            v_obs = self.v_obs_M
            vel_rp = self.params.constants.vel_rp

            U = ufl.Max((u_obs**2 + v_obs**2)**(1/2.0), 50.0)

            # Flotation Criterion
            H_flt = -rhow/rhoi * bed
            fl_ex = conditional(H <= H_flt, 1.0, 0.0)

            # Thickness Criterion
            m_d = conditional(H > 0, 1.0, 0.0)

            # Calculate surface gradient
            R_f = ((1.0 - fl_ex) * bed
                   + (fl_ex) * (-rhoi / rhow) * H)

            s_ = ufl.Max(H + R_f, 0)
            s = project(s_, self.Q)
            grads = (s.dx(0)**2.0 + s.dx(1)**2.0)**(1.0/2.0)

            # Calculate alpha, apply background, apply bound
            B2_ = ( (1.0 - fl_ex) * rhoi*g*H*grads/U
                    + (fl_ex) * a_bgd ) * m_d + (1.0-m_d) * a_bgd

            B2_tmp1 = ufl.Max(B2_, a_lb)
            B2_tmp2 = ufl.Min(B2_tmp1, a_ub)

            alpha = self.bdrag_to_alpha(B2_tmp2)
            self.alpha.assign(project(alpha, self.Qp))

        elif method == "constant":
            init_guess = self.params.inversion.initial_guess_alpha
            self.alpha.vector()[:] = init_guess
            self.alpha.vector().apply('insert')
        else:
            raise NotImplementedError(f"Don't have code for method {method}")
        function_update_state(self.alpha)

        write_diag = self.params.io.write_diagnostics
        if write_diag:
            diag_dir = self.params.io.diagnostics_dir
            phase_suffix = self.params.inversion.phase_suffix
            phase_name = self.params.inversion.phase_name
            inout.write_variable(self.alpha,
                                 self.params,
                                 name="alpha_init_guess",
                                 outdir=diag_dir,
                                 phase_name=phase_name,
                                 phase_suffix=phase_suffix)

    def mark_BCs(self):
        """
        Set up Facet Functions defining BCs

        If no bc_filename defined, check no BCs requested in TOML file (error),
        and check that the domain is periodic (warn)
        """
        # Do nothing if BCs aren't defined
        if self.params.mesh.bc_filename is None:
            assert len(self.params.bcs) == 0, \
                "Boundary Conditions [[BC]] defined but no bc_filename specified"

            if not self.params.mesh.periodic_bc:
                logging.warn("No BCs defined but mesh is not periodic?")

            self.ff = None

        else:
            # Read the facet function from a file containing a sparse MeshValueCollection
            self.ff = fice_mesh.get_ff_from_file(self.params, model=self, fill_val=0)

    def get_prior(self):
        """Get the prior based on params"""
        if self.params.inversion.delta_beta_gnd is not None:
            logging.info("Using Laplacian prior with flotation")
            return prior.Laplacian_flt
        else:
            logging.info("Using Laplacian prior without flotation")
            return prior.Laplacian

class PeriodicBoundary(SubDomain):
    def __init__(self, L):
        self.L = L
        super(PeriodicBoundary, self).__init__()

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        """
        Return True if on left or bottom boundary AND NOT on one of
        the two corners (0, L) and (L, 0)
        """
        return bool((near(x[0], 0) or near(x[1], 0))
                    and (not ((near(x[0], 0) and near(x[1], self.L))
                              or (near(x[0], self.L) and near(x[1], 0)))) and on_boundary)

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
