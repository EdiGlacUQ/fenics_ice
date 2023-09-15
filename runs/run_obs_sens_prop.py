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

from fenics_ice.backend import Function, HDF5File, project
from tlm_adjoint import reset_manager, set_manager, stop_manager, \
        configure_tlm, function_tlm, restore_manager,\
        EquationManager, start_manager

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import mpi4py.MPI as MPI  # noqa: N817
from pathlib import Path
import pickle
import numpy as np
import sys

from fenics_ice import model, solver, inout
from fenics_ice.model import interp_weights, interpolate
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
from ufl import split
from fenics_ice.solver import Amat_obs_action

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

# the function below is not in run_errorprop.py, but needed for obs sens
@restore_manager
def compute_tau(forward, u, m, dm):
    # this block of code will do the "forward" (calculation of velocity and cost function) once
    # and then find the jacobian of u in the direction needed
    set_manager(EquationManager(cp_method="none", cp_parameters={}))
    stop_manager()

    start_manager(tlm=True)
    configure_tlm((m, dm))
    forward(m)
    return function_tlm(u, (m, dm))        


def run_obs_sens_prop(config_file):

    # Read run config file
    params = ConfigParser(config_file)

    # Setup logging
    inout.setup_logging(params)
    inout.log_preamble("obssensprop", params)

    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    # Eigen decomposition params
    phase_eigen = params.eigendec.phase_name
    phase_suffix_e = params.eigendec.phase_suffix
    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file

    # Qoi forward params
    phase_time = params.time.phase_name
    phase_suffix_qoi = params.time.phase_suffix
    dqoi_h5file = params.io.dqoi_h5file

    if len(phase_suffix_e) > 0:
        lamfile = params.io.run_name + phase_suffix_e + '_eigvals.p'
        vecfile = params.io.run_name + phase_suffix_e + '_vr.h5'
    if len(phase_suffix_qoi) > 0:
        dqoi_h5file = params.io.run_name + phase_suffix_qoi + '_dQ_ts.h5'

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model
    mdl = model.model(mesh, input_data, params)

    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()
    mdl.bglen_from_data(mask_only=True)


    # Setup our solver object -- obs_sensitivity flag set to ensure caching
    slvr = solver.ssa_solver(mdl, mixed_space=params.inversion.dual, obs_sensitivity=True)

    # from errorprop -- this variable is not used in that fn
    #cntrl = slvr.get_control()[0]

    cntrl = slvr.get_control()
    space = slvr.get_control_space()
        
    # call to slvr.forward() needed to cache variables
    slvr.forward(cntrl)

    # Regularization operator using inversion delta/gamma values
    Prior = mdl.get_prior()
    reg_op = Prior(slvr, space)

    # Loads eigenvalues from file
    outdir_e = Path(outdir)/phase_eigen/phase_suffix_e
    with open(outdir_e/lamfile, 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)

    # Check if eigendecomposition successfully produced num_eig
    # or if some are NaN
    if np.any(np.isnan(lam)):
        raise RuntimeError("NaN eigenvalue(s)")

    # and eigenvectors from .h5 file
    eps = params.constants.float_eps
    W = []
    with HDF5File(MPI.COMM_WORLD, str(outdir_e/vecfile), 'r') as hdf5data:
        for i in range(nlam):
            w = Function(space)
            hdf5data.read(w, f'v/vector_{i}')

            # Test squared norm in prior == 1.0
            B_inv_w = Function(space, space_type="conjugate_dual")
            reg_op.action(w.vector(), B_inv_w.vector())
            norm_sq_in_prior = w.vector().inner(B_inv_w.vector())
            assert (abs(norm_sq_in_prior - 1.0) < eps)
            del B_inv_w

            W.append(w)

    D = np.diag(lam / (lam + 1))  # D_r Isaac 20

    # File containing dQoi_dCntrl (i.e. Jacobian of parameter to observable (Qoi))
    outdir_qoi = Path(outdir)/phase_time/phase_suffix_qoi
    hdf5data = HDF5File(MPI.COMM_WORLD, str(outdir_qoi/dqoi_h5file), 'r')

    dQ_cntrl = Function(space, space_type="conjugate_dual")

    run_length = params.time.run_length
    num_sens = params.time.num_sens
    t_sens = np.flip(np.linspace(run_length, 0, num_sens))

    # above this point there is very little difference with run_errorprop.py
    # -- exceptions are the defined function compute_tau() and 
    # -- the call to solver.forward() to generate interpolation matrix

    # this simply loads cached variables from solver.py which are cached in 
    #  the above forward() call
    assert hasattr(slvr,'_cached_Amat_vars'),\
        "Attempt to retrieve Amat from solver type without first caching"
    (P, u_std_local, v_std_local, interp_space) = \
                slvr._cached_Amat_vars

    # This is the matrix R in the definition of matrix A (separate for each u and v)
    Ru = spdiags(1.0 / (u_std_local ** 2),
                              0, P.shape[0], P.shape[0])
    Rv = spdiags(1.0 / (v_std_local ** 2),
                              0, P.shape[0], P.shape[0])

    dObsU = []
    dObsV = []
    
    # the following definitions are purely for visualisation
    #M_coords = mdl.M.tabulate_dof_coordinates()
    #vtx_M, wts_M = interp_weights(mdl.vel_obs['uv_obs_pts'],
    #                              M_coords, params.mesh.periodic_bc)
    #dObsU_M = []
    #dObsV_M = []

    for j in range(num_sens):

        # for each time level T, we have a Q_T, hence the loop
            
        hdf5data.read(dQ_cntrl, f'dQd{cntrl[0].name()}/vector_{j}')

        # the rest of this loop implements the same operations as 
        # lines 137-147 of run_errorprop.py, ie
        #
        #   (Gamma_{prior} - W D W^T) acting on (dQ/dm)
        #
        # P3 needed to be defined as it is because
        # P_vec = P2.vector() - P1.vector() led to errors later on (cannot recall specifics)
            
        tmp1 = np.asarray([w.vector().inner(dQ_cntrl.vector()) for w in W])
        tmp2 = np.dot(D, tmp1)

        P1 = Function(space)
        for tmp, w in zip(tmp2, W):
            P1.vector().axpy(tmp, w.vector())

        P2 = Function(space)
        reg_op.inv_action(dQ_cntrl.vector(), P2.vector())

        P3 = Function(space)
        P3.vector()[:] = P2.vector()[:] - P1.vector()[:]
        

        # retaining the following code causes a tlm_adjoint runtime error in the 2nd execution of the loop 
        #  so the code is packaged within a decorated function
        # reset_manager()
        # stop_manager(tlm=False)
        # configure_tlm((cntrl, P3))
        # slvr.forward(cntrl)
        # tau = function_tlm(slvr.U, (cntrl, P3))

        # tau is  d(U,V)/dm * (Gamma_{prior} - W D W^T) * (dQ/dm), or
        #         d(U,V)/dm * P3
        tau = compute_tau(slvr.forward, slvr.U, cntrl, P3)

        # tau is in the space of U (right?)
        tauu, tauv = split(tau)

        #tauuQ = project(tauu,mdl.Q)
        #tauvQ = project(tauv,mdl.Q)

        #if ( j==(num_sens-1) ):
        # x    = mesh.coordinates()[:,0]
        # y    = mesh.coordinates()[:,1]
        # t    = mesh.cells()
        # cmap_div='RdBu'
        # V = tauvQ
        # v   = V.compute_vertex_values(mesh)
        # plt.tricontourf(x, y, t, v, cmap=plt.get_cmap(cmap_div))
        # plt.show()




        # this block of code then implements A.tau
        # but it needs to be made negative..

        dobsu = Amat_obs_action(P, Ru, tauu, interp_space)
        dobsv = Amat_obs_action(P, Rv, tauv, interp_space)
        dObsU.append(dobsu)
        dObsV.append(dobsv)

        # this iqs simply interpolation for later visualisation
        # this could be a point of error.. but I don't know
        # how to test this by visualising dObsU, dObsV!
        #dObsU_M.append(Function(mdl.M, static=True))
        #dObsV_M.append(Function(mdl.M, static=True))
        #dObsU_M[j].vector()[:] = interpolate(dobsu, vtx_M, wts_M)
        #dObsV_M[j].vector()[:] = interpolate(dobsv, vtx_M, wts_M)




# below is some plotting commands, will not be used her but in another script
    
#    V = mdl.u_obs_M
#    ax  = fig.add_subplot(221)
#    ax.set_aspect('equal')
#    v   = V.compute_vertex_values(mesh)
#    ticks = np.linspace(10,30,3)
#    c = ax.tricontourf(x, y, t, v, cmap=plt.get_cmap(cmap_div))
#    cbar = plt.colorbar(c, ticks=ticks, pad=0.05)
#
#    V = mdl.v_obs_M
#    ax  = fig.add_subplot(222)
#    ax.set_aspect('equal')
#    v   = V.compute_vertex_values(mesh)
#    ticks = np.linspace(10,30,3)
#    c = ax.tricontourf(x, y, t, v, cmap=plt.get_cmap(cmap_div))
#    cbar = plt.colorbar(c, ticks=ticks, pad=0.05)
#
#    numlev = 20
#    tick_options = {'axis':'both','which':'both','bottom':False,
#        'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}
#
#    V = dObsU_M[-1]
#    ax  = fig.add_subplot(223)
#    ax.set_aspect('equal')
#    v   = V.compute_vertex_values(mesh)
#    minv = np.min(v)
#    maxv = np.max(v)
#    levels = np.linspace(minv,maxv,numlev)
#    ticks = np.linspace(minv,maxv,3)
#    ax.tick_params(**tick_options)
#    ax.text(0.05, 0.95, 'a', transform=ax.transAxes,
#    fontsize=13, fontweight='bold', va='top')
#    c = ax.tricontourf(x, y, t, v, levels = levels, cmap=plt.get_cmap(cmap_div))
#    cbar = plt.colorbar(c, ticks=ticks, pad=0.05)

#    V = dObsV_M[-1]
#    ax  = fig.add_subplot(224)
#    ax.set_aspect('equal')
#    v   = V.compute_vertex_values(mesh)
#    minv = np.min(v)
#    maxv = np.max(v)
#    levels = np.linspace(minv,maxv,numlev)
#    ticks = np.linspace(minv,maxv,3)
#    ax.tick_params(**tick_options)
#    ax.text(0.05, 0.95, 'a', transform=ax.transAxes,
#    fontsize=13, fontweight='bold', va='top')
#    c = ax.tricontourf(x, y, t, v, levels = levels, cmap=plt.get_cmap(cmap_div))
#    cbar = plt.colorbar(c, ticks=ticks, pad=0.05)


               

    # Save plots in diagnostics
    phase_sens = params.obs_sens.phase_name
    phase_suffix_sens = params.obs_sens.phase_suffix
    diag_dir = Path(params.io.diagnostics_dir)/phase_sens/phase_suffix_sens

    with open( os.path.join(diag_dir, 'vel_sens.npy'), "wb" ) as sensfile:
        np.save(sensfile, [dObsU, dObsV])
        sensfile.close()
    #    pickle.dump( [sigma, t_sens], sigfile)
    #with open( os.path.join(outdir_err, sigma_prior_file), "wb" ) as sigpfile:
    #    pickle.dump( [sigma_prior, t_sens], sigpfile)



if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_obs_sens_prop(sys.argv[1])
