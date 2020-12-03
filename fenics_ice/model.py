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

from fenics import *
from dolfin import *
import ufl
import numpy as np
from pathlib import Path
import scipy.spatial.qhull as qhull
from fenics_ice import inout
from fenics_ice import mesh as fice_mesh
from numpy.random import randn
import logging

log = logging.getLogger("fenics_ice")

class model:

    def __init__(self, mesh_in, input_data, param_in, init_fields=True,
                 init_vel_obs=True):

        # Initiate parameters
        self.params = param_in
        self.input_data = input_data
        self.solvers = []
        self.parallel = MPI.size(mesh_in.mpi_comm()) > 1

        # Generate Domain and Function Spaces
        self.mesh = mesh_in
        self.nm = FacetNormal(self.mesh)
        self.Q = FunctionSpace(self.mesh, 'Lagrange', 1)

        self.M = FunctionSpace(self.mesh, 'DG', 0)
        self.RT = FunctionSpace(self.mesh, 'RT', 1)

        # Based on IsmipC: alpha, beta, and U are periodic.
        if not self.params.mesh.periodic_bc:
            self.Qp = self.Q
            self.V = VectorFunctionSpace(self.mesh, 'Lagrange', 1, dim=2)
        else:
            self.Qp = fice_mesh.get_periodic_space(self.params, self.mesh, dim=1)
            self.V = fice_mesh.get_periodic_space(self.params, self.mesh, dim=2)

        # Default velocity mask and Beta fields
        self.def_vel_mask()
        self.def_B_field()
        self.def_lat_dirichletbc()

        self.mark_BCs()
        if init_fields:
            self.init_fields_from_data()

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

        self.bed = self.field_from_data("bed", self.Q, static=True)
        self.bmelt = self.field_from_data("bmelt", self.M, default=0.0, static=True)
        self.smb = self.field_from_data("smb", self.M, default=0.0, static=True)
        self.H_np = self.field_from_data("thick", self.M, min_val=min_thick)

        self.H = self.H_np.copy(deepcopy=True)
        self.H.rename("thick_H", "")

        self.gen_surf()  # surf = bed + thick

    def def_vel_mask(self):
        self.mask_vel_M = project(Constant(0.0), self.M)

    def def_B_field(self):
        """Define beta field from constants in config file"""
        A = self.params.constants.A
        n = self.params.constants.glen_n
        self.beta = project(self.bglen_to_beta(A**(-1.0/n)), self.Qp)
        self.beta_bgd = project(self.bglen_to_beta(A**(-1.0/n)), self.Qp)
        self.beta.rename('beta', 'a Function')
        self.beta_bgd.rename('beta_bgd', 'a Function')

    def def_lat_dirichletbc(self):
        """Homogenous dirichlet conditions on lateral boundaries"""
        self.latbc = Constant([0.0, 0.0])

    def field_from_data(self, name, space, **kwargs):
        """Interpolate a named field from input data"""
        return self.input_data.interpolate(name, space, **kwargs)

    def alpha_from_data(self):
        """Get alpha field from initial input data (run_momsolve only)"""
        self.alpha = self.input_data.interpolate("alpha", self.Qp)

    def bglen_from_data(self):
        """Get bglen field from initial input data"""
        self.bglen = self.input_data.interpolate("Bglen", self.Q)

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

        self.beta_bgd.rename('beta_bgd', 'a Function')

    def init_beta(self, beta, pert=False):
        """
        Define the beta field from input

        Optionally perturb the field slightly to prevent zero gradient
        on first step of beta inversion.
        """
        self.beta_bgd = project(beta, self.Qp)
        self.beta = project(beta, self.Qp)
        if pert:
            # Perturbed field for nonzero gradient at first step of inversion
            bv = self.beta.vector().get_local()
            pert_vec = 0.001*bv*randn(bv.size)
            self.beta.vector().set_local(bv + pert_vec)
            self.beta.vector().apply('insert')

        self.beta.rename('beta', 'a Function')
        self.beta_bgd.rename('beta_bgd', 'a Function')

    def vel_obs_from_data(self):
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
        # TODO - this will not handle extrapolation/missing data
        # nicely - unfound simplex are returned '-1' which takes the last
        # tri.simplices...
        def interp_weights(xy, uv, d=2):
            """Compute the nearest vertices & weights (for reuse)"""
            tri = qhull.Delaunay(xy)
            simplex = tri.find_simplex(uv)

            if not np.all(simplex >= 0):
                if not self.params.mesh.periodic_bc:
                    log.error("Some points missing in interpolation "
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

        # Grab coordinates of both Lagrangian & DG function spaces
        # and compute (once) the interpolating arrays
        Q_coords = self.Q.tabulate_dof_coordinates()
        M_coords = self.M.tabulate_dof_coordinates()

        vtx_Q, wts_Q = interp_weights(self.uv_obs_pts, Q_coords)
        vtx_M, wts_M = interp_weights(self.uv_obs_pts, M_coords)

        # Define new functions to hold results
        self.u_obs_Q = Function(self.Q, name="u_obs", static=True)
        self.v_obs_Q = Function(self.Q, name="v_obs", static=True)
        self.u_std_Q = Function(self.Q, name="u_std", static=True)
        self.v_std_Q = Function(self.Q, name="v_std", static=True)
        # self.mask_vel_Q = Function(self.Q)

        self.u_obs_M = Function(self.M, name="u_obs", static=True)
        self.v_obs_M = Function(self.M, name="v_obs", static=True)
        # self.u_std_M = Function(self.M)
        # self.v_std_M = Function(self.M)
        self.mask_vel_M = Function(self.M, name="mask_vel", static=True)

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
        self.surf._Function_static__ = True
        self.surf._Function_checkpoint__ = False
        self.surf.rename("surf", "")

    def gen_alpha(self, a_bgd=500.0, a_lb=1e2, a_ub=1e4):
        """Generate initial guess for alpha (slip coeff)"""
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

        sl = self.params.ice_dynamics.sliding_law
        if sl == 'linear':
            alpha = sqrt(B2_tmp2)
        elif sl == 'weertman':
            N = (1-fl_ex)*(H*rhoi*g + ufl.Min(bed, 0.0)*rhow*g)
            U_mag = sqrt(u_obs**2 + v_obs**2 + vel_rp**2)
            alpha = (1-fl_ex)*sqrt(B2_tmp2 * ufl.Max(N, 0.01)**(-1.0/3.0) * U_mag**(2.0/3.0))

        self.alpha = project(alpha, self.Qp)
        self.alpha.rename('alpha', 'a Function')

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
