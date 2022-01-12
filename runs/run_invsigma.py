
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

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import pickle
import numpy as np

from dolfin import *
from tlm_adjoint.fenics import *

from fenics_ice import model, solver, prior, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser

# import matplotlib as mpl
# mpl.use("Agg")
# import matplotlib.pyplot as plt

def patch_fun(mesh_in, params):
    """
    Create 'patches' of cells for invsigma calculation. Takes a random sample of
    cell midpoints via DG0 space, which become patch centrepoints. Then each DOF
    is assigned to its nearest centrepoint.

    Returns a DG0 function where each cell is numbered by its assigned patch.
    """
    import random
    from scipy.spatial import KDTree

    comm = MPI.comm_world
    rank = MPI.rank(comm)
    root = rank == 0

    # Test DG function
    # DG0 gives triangle centroids
    dg = FunctionSpace(mesh_in, 'DG', 0)
    dg_fun = Function(dg)

    # Get a random sample of cells (root bcast)
    ncells = dg_fun.vector().size()

    # Ratio of n_patches to n_cells
    if params.inv_sigma.npatches is not None:
        ntgt = param.inv_sigma.npatches
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

    comm = MPI.comm_world
    rank = comm.rank

    # Read run config file
    params = ConfigParser(config_file)

    # Setup logging
    log = inout.setup_logging(params)
    inout.log_preamble("inv sigma", params)

    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    eigendir = outdir
    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file
    threshlam = params.eigendec.eigenvalue_thresh

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model (only need alpha & beta though)
    mdl = model.model(mesh, input_data, params, init_fields=True)

    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()

    # Setup our solver object
    slvr = solver.ssa_solver(mdl, mixed_space=params.inversion.dual)

    cntrl = slvr.get_control()[0]
    space = slvr.get_control_space()

    # sigma_old, sigma_prior_old = [Function(space) for i in range(3)]
    x, y, z = [Function(space) for i in range(3)]
    # Regularization operator using inversion delta/gamma values
    Prior = mdl.get_prior()
    reg_op = Prior(slvr, space)

    # Load the eigenvalues
    with open(os.path.join(eigendir, lamfile), 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)

    # Check if eigendecomposition successfully produced num_eig
    # or if some are NaN
    if np.any(np.isnan(lam)):
        nlam = np.argwhere(np.isnan(lam))[0][0]
        lam = lam[:nlam]

    # Read in the eigenvectors and check they are normalised
    # w.r.t. the prior (i.e. the B matrix in our GHEP)
    eps = params.constants.float_eps
    W = []
    with HDF5File(comm,
                  os.path.join(eigendir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            w = Function(space)
            hdf5data.read(w, f'v/vector_{i}')

            print(f"Getting eigenvector {i} of {nlam}")
            # # Test norm in prior == 1.0
            # reg_op.action(w.vector(), y.vector())
            # norm_in_prior = w.vector().inner(y.vector())
            # assert (abs(norm_in_prior - 1.0) < eps)

            W.append(w)

    # Which eigenvalues are larger than our threshold?
    pind = np.flatnonzero(lam > threshlam)
    lam = lam[pind]
    W = [W[i] for i in pind]

    # this is a diagonal matrix but we only ever address it element-wise
    # bit of a waste of space.
    D = np.diag(lam / (lam + 1))

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
        log.warning('Negative value(s) of sigma encountered.'
                    'Examine the range of eigenvalues and check if '
                    'the threshlam paramater is set appropriately.')

    # # Previous approach for comparison
    # #####################################

    # # Isaac Eq. 20
    # # P2 = prior
    # # P1 = WDW
    # # Note - don't think we're considering the cross terms
    # # in the posterior covariance.
    # # TODO - this isn't particularly well parallelised - can it be improved?
    # neg_flag = 0
    # for j in range(space.dim()):

    #     # Who owns this DOF?
    #     own_idx = y.vector().owns_index(j)
    #     ownership = np.where(comm.allgather(own_idx))[0]
    #     assert len(ownership) == 1
    #     idx_root  = ownership[0]

    #     # Prior (P2)
    #     y.vector().zero()
    #     y.vector().vec().setValue(j, 1.0)
    #     y.vector().apply('insert')
    #     reg_op.inv_action(y.vector(), x.vector())
    #     P2 = x

    #     # WDW (P1) ~ lam * V_r**2
    #     tmp2 = np.asarray([D[i, i] * w.vector().vec().getValue(j) for i, w in enumerate(W)])
    #     tmp2 = comm.bcast(tmp2, root=idx_root)

    #     P1 = Function(space)
    #     for tmp, w in zip(tmp2, W):
    #         P1.vector().axpy(tmp, w.vector())

    #     P_vec = P2.vector() - P1.vector()

    #     # Extract jth component & save
    #     # TODO why does this need to be communicated here? surely owning proc
    #     # just inserts?
    #     dprod = comm.bcast(P_vec.vec().getValue(j), root=idx_root)
    #     dprod_prior = comm.bcast(P2.vector().vec().getValue(j), root=idx_root)

    #     if dprod < 0:
    #         log.warning(f'WARNING: Negative Sigma: {dprod}')
    #         log.warning('Setting as Zero and Continuing.')
    #         neg_flag = 1
    #         continue

    #     sigma_old.vector().vec().setValue(j, np.sqrt(dprod))
    #     sigma_prior_old.vector().vec().setValue(j, np.sqrt(dprod_prior))

    # sigma_old.vector().apply("insert")
    # sigma_prior_old.vector().apply("insert")

    # For testing - whole thing at once:
    # wdw = (np.matrix(W) * np.matrix(D) * np.matrix(W).T)
    # wdw[:,0] == P1 for j = 0

    # if neg_flag:
    #     log.warning('Negative value(s) of sigma encountered.'
    #                 'Examine the range of eigenvalues and check if '
    #                 'the threshlam paramater is set appropriately.')

    # Write sigma & sigma_prior to files
    # sigma_var_name = "_".join((cntrl.name(), "sigma"))
    # sigma_prior_var_name = "_".join((cntrl.name(), "sigma_prior"))

    # sigma_old.rename(sigma_var_name, "")
    # sigma_prior_old.rename(sigma_prior_var_name, "")

    # inout.write_variable(sigma_old, params,
    #                      name=sigma_var_name+"_old")
    # inout.write_variable(sigma_prior_old, params,
    #                      name=sigma_prior_var_name+"_old")

    for i, name in enumerate(cntrl_names):
        sigmas[i].rename("sigma_"+name, "")
        sigma_priors[i].rename("sigma_prior_"+name, "")

        inout.write_variable(sigmas[i], params)
        inout.write_variable(sigma_priors[i], params)

    mdl.cntrl_sigma = sigmas
    mdl.cntrl_sigma_prior = sigma_priors
    return mdl


if __name__ == "__main__":
    stop_annotating()

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_invsigma(sys.argv[1])
