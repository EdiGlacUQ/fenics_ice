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

from fenics_ice.backend import FiniteElement, Function, FunctionSpace, \
    HDF5File, TestFunction, assemble, assign, inner, dx

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
import pickle
import numpy as np
import sys

from fenics_ice import model, solver, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser


def patch_fun(mesh_in, params):
    """
    Create 'patches' of cells for invsigma calculation. Takes a random sample of
    cell midpoints via DG0 space, which become patch centrepoints. Then each DOF
    is assigned to its nearest centrepoint.

    Returns a DG0 function where each cell is numbered by its assigned patch.
    """
    import random
    from scipy.spatial import KDTree

    comm = mesh_in.mpi_comm()
    rank = comm.rank
    root = rank == 0

    # Test DG function
    # DG0 gives triangle centroids
    dg = FunctionSpace(mesh_in, 'DG', 0)
    dg_fun = Function(dg)

    # Get a random sample of cells (root bcast)
    ncells = dg_fun.vector().size()

    # Ratio of n_patches to n_cells
    if params.inv_sigma.npatches is not None:
        ntgt = params.inv_sigma.npatches
    else:
        ntgt = int(np.floor(ncells * params.inv_sigma.patch_downscale))

    if root:
        if params.constants.random_seed is not None:
            random.seed(params.constants.random_seed)

        tgt_cells = random.sample(range(ncells), ntgt)
        tgt_cells.sort()
    else:
        tgt_cells = None

    # Send to other procs
    tgt_cells = comm.bcast(tgt_cells, root=0)

    # Each compute own range
    my_min, my_max = dg.dofmap().ownership_range()
    my_max -= 1
    my_tgt_cells = tgt_cells[np.searchsorted(tgt_cells, my_min):
                             np.searchsorted(tgt_cells, my_max, side='right')]

    # Get the requested local dofs
    dg_gdofs = dg.dofmap().tabulate_local_to_global_dofs()  # all global dofs
    dg_ldofs = np.arange(0, dg_gdofs.size)  # all local dofs

    dg_gdof_idx = np.argsort(dg_gdofs)  # idx which sorts gdofs
    dg_gdofs_sorted = dg_gdofs[dg_gdof_idx]  # sorted gdofs
    dg_ldofs_sorted = dg_ldofs[dg_gdof_idx]  # 'sorted' ldofs

    # Search our tgt_cells (gdofs) in sorted gdof list
    tgt_local_idx = np.searchsorted(dg_gdofs_sorted, my_tgt_cells)
    tgt_local = np.take(dg_ldofs_sorted, tgt_local_idx, mode='raise')
    tgt_global = np.take(dg_gdofs_sorted, tgt_local_idx, mode='raise')

    assert np.all(tgt_global == my_tgt_cells),\
        "Logic error - failed to find all tgt global dofs"

    # Get the cell midpoints for local targets
    my_tgt_cell_mids = dg.tabulate_dof_coordinates()[tgt_local]
    # and broadcast to all
    tgt_cell_mids = np.vstack(comm.allgather(my_tgt_cell_mids))

    # KDTree search to find nearest tgt midpoint
    tree = KDTree(tgt_cell_mids)
    dist, nearest = tree.query(dg.tabulate_dof_coordinates())

    dg_fun = Function(dg)
    dg_fun.vector()[:] = nearest
    dg_fun.vector().apply("insert")

    return dg_fun, ntgt


def run_invsigma(config_file):
    """Compute control sigma values from eigendecomposition"""
    # Read run config file
    params = ConfigParser(config_file)

    # Setup logging
    log = inout.setup_logging(params)
    inout.log_preamble("inv sigma", params)

    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    # Eigen decomposition params
    phase_suffix_e = params.eigendec.phase_suffix
    eigendir = Path(outdir)/params.eigendec.phase_name/phase_suffix_e
    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file

    if len(phase_suffix_e) > 0:
        lamfile = params.io.run_name + phase_suffix_e + '_eigvals.p'
        vecfile = params.io.run_name + phase_suffix_e + '_vr.h5'

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)
    comm = mesh.mpi_comm()

    # Define the model (only need alpha & beta though)
    mdl = model.model(mesh, input_data, params, init_fields=True)

    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()
    mdl.bglen_from_data(mask_only=True)

    # Setup our solver object
    slvr = solver.ssa_solver(mdl, mixed_space=params.inversion.dual)

    space = slvr.get_control_space()

    x, y, z = [Function(space) for i in range(3)]
    # Regularization operator using inversion delta/gamma values
    Prior = mdl.get_prior()
    reg_op = Prior(slvr, space)

    # Loads eigenvalues from file
    with open(os.path.join(eigendir, lamfile), 'rb') as ff:
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
    with HDF5File(comm,
                  os.path.join(eigendir, vecfile), 'r') as hdf5data:
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

    # TODO make this a model method
    cntrl_names = []
    if params.inversion.alpha_active:
        cntrl_names.append("alpha")
    if params.inversion.beta_active:
        cntrl_names.append("beta")
    dual = params.inversion.dual

    ############################################
    # Isaac Eq. 20
    # P2 = prior
    # P1 = WDW
    # Note - don't think we're considering the cross terms
    # in the posterior covariance.

    # Generate patches of cells for computing invsigma
    clust_fun, npatches = patch_fun(mesh, params)

    # Create standard & mixed DG spaces
    dg_space = FunctionSpace(mesh, 'DG', 0)
    if(dual):
        dg_el = FiniteElement("DG", mesh.ufl_cell(), 0)
        mixedEl = dg_el * dg_el
        dg_out_space = FunctionSpace(mesh, mixedEl)
    else:
        dg_out_space = dg_space

    sigmas = [Function(dg_space) for i in range(len(cntrl_names))]
    sigma_priors = [Function(dg_space) for i in range(len(cntrl_names))]

    indic_1 = Function(dg_space)
    indic = Function(dg_out_space)

    test = TestFunction(space)

    neg_flag = 0
    for i in range(npatches):

        print(f"Working on patch {i+1} of {npatches}")

        # Create DG indicator function for patch i
        indic_1.vector()[:] = (clust_fun.vector()[:] == i).astype(int)
        indic_1.vector().apply("insert")

        # Loop alpha & beta as appropriate
        for j in range(len(cntrl_names)):

            if(dual):
                indic.vector()[:] = 0.0
                indic.vector().apply("insert")
                assign(indic.sub(j), indic_1)
            else:
                assign(indic, indic_1)

            clust_lump = assemble(inner(indic, test)*dx)
            patch_area = clust_lump.sum()  # Duplicate work here...

            clust_lump /= patch_area

            # Prior variance
            reg_op.inv_action(clust_lump, x.vector())
            cov_prior = x.vector().inner(clust_lump)

            # P_i^T W D W^T P_i
            # P_i is clust_lump
            # P_i^T has dims [1 x M], W has dims [M x N]
            # where N is num eigs & M is size of ev function space
            PiW = np.asarray([clust_lump.inner(w.vector()) for w in W])

            # PiW & PiWD are [1 x N]
            PiWD = PiW * D.diagonal()
            # PiWDWPi, [1 x N] * [N x 1]
            PiWDWPi = np.inner(PiWD, PiW)  # np.inner OK here because already parallel reduced

            cov_reduction = PiWDWPi
            cov_post = cov_prior - cov_reduction

            if cov_post < 0:
                log.warning(f'WARNING: Negative Sigma: {cov_post}')
                log.warning('Setting as Zero and Continuing.')
                neg_flag = 1
                continue

            # NB: "+=" here but each DOF will only be contributed to *once*
            # Essentially we are constructing the sigmas functions from
            # non-overlapping patches.
            sigmas[j].vector()[:] += indic_1.vector()[:] * np.sqrt(cov_post)
            sigmas[j].vector().apply("insert")

            sigma_priors[j].vector()[:] += indic_1.vector()[:] * np.sqrt(cov_prior)
            sigma_priors[j].vector().apply("insert")

    if neg_flag:
        log.warning('Negative value(s) of sigma encountered')

    for i, name in enumerate(cntrl_names):
        sigmas[i].rename("sigma_"+name, "")
        sigma_priors[i].rename("sigma_prior_"+name, "")

        phase_suffix_sigma = params.inv_sigma.phase_suffix

        inout.write_variable(sigmas[i], params,
                             outdir=outdir,
                             phase_name=params.inv_sigma.phase_name,
                             phase_suffix=phase_suffix_sigma)
        inout.write_variable(sigma_priors[i], params,
                             outdir=outdir,
                             phase_name=params.inv_sigma.phase_name,
                             phase_suffix=phase_suffix_sigma)

    mdl.cntrl_sigma = sigmas
    mdl.cntrl_sigma_prior = sigma_priors
    return mdl


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_invsigma(sys.argv[1])
