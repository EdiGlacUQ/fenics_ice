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

from fenics_ice.backend import Function, HDF5File, MPI, project

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import pickle
from pathlib import Path

from fenics_ice import model, solver, prior, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
from numpy import random

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def run_sample(config_file):

    # Read run config file
    params = ConfigParser(config_file)
    log = inout.setup_logging(params)
    inout.log_preamble("errorprop", params)

    ssize = params.sample.sample_size
    sample_posterior = params.sample.sample_posterior
    alpha_active = params.inversion.alpha_active
    beta_active = params.inversion.beta_active
    log = inout.setup_logging(params)
    inout.log_preamble("errorprop", params)

    phase_name_sample = params.sample.phase_name
    phase_suffix_sample = params.sample.phase_suffix

    outdir = params.io.output_dir
    diag_dir = params.io.diagnostics_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    #Eigen value params
    phase_name_e = params.eigendec.phase_name
    phase_suffix_e = params.eigendec.phase_suffix
    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file
    threshlam = params.eigendec.eigenvalue_thresh

    # Qoi forward params
    phase_time = params.time.phase_name
    phase_suffix_qoi = params.time.phase_suffix
    dqoi_h5file = params.io.dqoi_h5file

    if len(phase_suffix_e) > 0:
        lamfile = params.io.run_name + phase_suffix_e + '_eigvals.p'
        vecfile = params.io.run_name + phase_suffix_e + '_vr.h5'

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model
    mdl = model.model(mesh, input_data, params)

    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()
    mdl.bglen_from_data(mask_only=True)

    # Setup our solver object
    slvr = solver.ssa_solver(mdl, mixed_space=params.inversion.dual)

    cntrl = slvr.get_control()[0]
    space = slvr.get_control_space()

    # Regularization operator using inversion delta/gamma values
    Prior = mdl.get_prior()
    reg_op = Prior(slvr, space)

    if (sample_posterior):

        # Loads eigenvalues from file
        outdir_e = Path(outdir)/phase_name_e/phase_suffix_e
        with open(outdir_e/lamfile, 'rb') as ff:
            eigendata = pickle.load(ff)
            lam = eigendata[0].real.astype(np.float64)
            nlam = len(lam)


        # Check if eigendecomposition successfully produced num_eig
        # or if some are NaN
        if np.any(np.isnan(lam)):
            nlam = np.argwhere(np.isnan(lam))[0][0]
            lam = lam[:nlam]

        # and eigenvectors from .h5 file
        eps = params.constants.float_eps
        W = []

        max_lam = params.sample.num_eigenvals
        if (max_lam > 0):
            lam = lam[:max_lam] 

        y = Function(space)
        with HDF5File(MPI.comm_world, str(outdir_e/vecfile), 'r') as hdf5data:
            for i in range(len(lam)):
                w = Function(space)
                hdf5data.read(w, f'v/vector_{i}')

                # Test norm in prior == 1.0
                reg_op.action(w.vector(), y.vector())
                norm_in_prior = w.vector().inner(y.vector())
                assert (abs(norm_in_prior - 1.0) < eps)

                W.append(w)


      # take only the largest eigenvalues
        pind = np.flatnonzero(lam > threshlam)
        lam = lam[pind]
        nlam = len(lam)
        W = [W[i] for i in pind]

### ABOVE THIS POINT CODE IS BORROWED FROM RUN_ERRORPROP.PY

        D = np.diag(1 / np.sqrt(lam + 1) - 1)  

    x, z, zm = [Function(space) for i in range(3)]
    if (ssize>1):
        zstd = Function(space)
    if (sample_posterior):
        a, am = [Function(space) for i in range(2)]
        if (ssize>1):
            astd = Function(space)

    shp = np.shape(z.vector().get_local())

    zm.vector().zero()
    zm.vector().apply("insert")
    if (ssize>1):
        zstd.vector().zero()
        zstd.vector().apply("insert")
    
    if (sample_posterior):
        am.vector().zero()
        am.vector().apply("insert")
        y.vector().zero()
        y.vector().apply("insert")
        if (ssize>1):
            astd.vector().zero()
            astd.vector().apply("insert")

    np.random.seed()
    for i in range(params.sample.sample_size):

        x.vector().set_local(random.normal(np.zeros(shp),  # N
                         np.ones(shp),shp))
        x.vector().apply("insert")
	  
        reg_op.sqrt_inv_action(x.vector(),z.vector())  # Gamma 1/2 N

        zm.vector().set_local(zm.vector().get_local() + z.vector().get_local()/float(ssize))
        zm.vector().apply("insert")
        if (ssize>1):
            zstd.vector().set_local(zstd.vector().get_local() + z.vector().get_local()**2/float(ssize))
            zstd.vector().apply("insert")

        if (sample_posterior):
            reg_op.sqrt_action(x.vector(),y.vector())  # Gamma -1/2 N

            tmp1 = np.asarray([w.vector().inner(y.vector()) for w in W])
            tmp2 = np.dot(D,tmp1)

            P1 = Function(space)
            for ind in range(len(tmp2)):
                P1.vector().axpy(tmp2[ind],W[ind].vector())

            a.vector().set_local(z.vector().get_local() + P1.vector())
            a.vector().apply("insert")

            am.vector().set_local(am.vector().get_local() + a.vector().get_local()/float(ssize))
            am.vector().apply("insert")
            if (ssize>1):
                astd.vector().set_local(astd.vector().get_local() + a.vector().get_local()**2/float(ssize))
                astd.vector().apply("insert")

    if (ssize>1):
        zstd.vector().set_local(np.sqrt(zstd.vector().get_local() - zm.vector().get_local()**2))   
        zstd.vector().apply("insert")
     
    if (sample_posterior):
        if (ssize>1):
            astd.vector().set_local(np.sqrt(astd.vector().get_local() - am.vector().get_local()**2))
            astd.vector().apply("insert")

    if params.inversion.dual:
        alpha_prior_sample_mean = project(zm[0], slvr.Qp)
        beta_prior_sample_mean = project(zm[1], slvr.Qp)
        if (ssize>1):
            alpha_prior_sample_std = project(zstd[0], slvr.Qp)
            beta_prior_sample_std = project(zstd[1], slvr.Qp)
    
        if (sample_posterior):
            alpha_post_sample_mean = project(am[0], slvr.Qp)
            beta_post_sample_mean = project(am[1], slvr.Qp)
            if (ssize>1):
                alpha_post_sample_std = project(astd[0], slvr.Qp)
                beta_post_sample_std = project(astd[1], slvr.Qp)

    elif alpha_active:
        alpha_prior_sample_mean = zm
        if (ssize>1):
            alpha_prior_sample_std = zstd

        if (sample_posterior):
            alpha_post_sample_mean = am
            if (ssize>1):
                alpha_post_sample_std = astd

    elif beta_active:
        beta_prior_sample_mean = zm
        if (ssize>1):
            beta_prior_sample_std = zstd

        if (sample_posterior):
            beta_post_sample_mean = am
            if (ssize>1):
                beta_post_sample_std = astd

    if ((alpha_active or params.inversion.dual) and params.sample.sample_alpha):
        inout.write_variable(alpha_prior_sample_mean, params, name="alpha_prior_sample_mean_"+str(ssize), 
                          outdir=diag_dir,
                          phase_name=phase_name_sample, 
                          phase_suffix=phase_suffix_sample)
        if (ssize>1):
            inout.write_variable(alpha_prior_sample_std, params, name="alpha_prior_sample_stdev_"+str(ssize), 
                          outdir=diag_dir,
                          phase_name=phase_name_sample, 
                          phase_suffix=phase_suffix_sample)
        if (sample_posterior):
            inout.write_variable(alpha_post_sample_mean, params, name="alpha_posterior_sample_mean_"+str(ssize), 
                          outdir=diag_dir,
                          phase_name=phase_name_sample, 
                          phase_suffix=phase_suffix_sample)
            if (ssize>1):
                inout.write_variable(alpha_post_sample_std, params, name="alpha_posterior_sample_stdev_"+str(ssize), 
                          outdir=diag_dir,
                          phase_name=phase_name_sample, 
                          phase_suffix=phase_suffix_sample)

    if ((beta_active or params.inversion.dual) and params.sample.sample_alpha):
        inout.write_variable(beta_prior_sample_mean, params, name="beta_prior_sample_mean_"+str(ssize), 
                          outdir=diag_dir,
                          phase_name=phase_name_sample, 
                          phase_suffix=phase_suffix_sample)
        if (ssize>1):
            inout.write_variable(beta_prior_sample_std, params, name="beta_prior_sample_stdev_"+str(ssize), 
                          outdir=diag_dir,
                          phase_name=phase_name_sample, 
                          phase_suffix=phase_suffix_sample)
        if (sample_posterior):
            inout.write_variable(beta_post_sample_mean, params, name="beta_posterior_sample_mean_"+str(ssize), 
                          outdir=diag_dir,
                          phase_name=phase_name_sample, 
                          phase_suffix=phase_suffix_sample)
            if (ssize>1):
                inout.write_variable(beta_post_sample_std, params, name="beta_posterior_sample_stdev_"+str(ssize), 
                          outdir=diag_dir,
                          phase_name=phase_name_sample, 
                          phase_suffix=phase_suffix_sample)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_sample(sys.argv[1])
