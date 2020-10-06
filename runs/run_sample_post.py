import sys

import os
import argparse
from dolfin import *
from tlm_adjoint import *

from fenics_ice import model, solver, prior, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
import fenics_ice.fenics_util as fu
from numpy import random
from pathlib import Path



import matplotlib as mpl
#mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import pickle
from petsc4py import PETSc
from IPython import embed
import logging as log

def run_sample_post(config_file):

    print("GOT HERE")

    #Read run config file

    assert MPI.size(MPI.comm_world) == 1, "Run this stage in serial!"

    # Read run config file
    params = ConfigParser(config_file)

    plotdir = Path(os.environ['FENICS_ICE_BASE_DIR']) / 'example_cases' / params.io.run_name / 'plots'
    plotdir.mkdir(parents=True, exist_ok=True)

    log = inout.setup_logging(params)
    inout.log_preamble("errorprop", params)

    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    lamfile = params.io.eigenvalue_file
    vecfile = params.io.eigenvecs_file
    threshlam = params.eigendec.eigenvalue_thresh

    # Get model mesh
    mesh = fice_mesh.get_mesh(params)

    # Define the model
    mdl = model.model(mesh, input_data, params)

    # Load alpha/beta fields
    mdl.alpha_from_inversion()
    mdl.beta_from_inversion()


    # Regularization operator using inversion delta/gamma values
    # TODO - this won't handle dual inversion case
    if params.inversion.alpha_active:
        delta = params.inversion.delta_alpha
        gamma = params.inversion.gamma_alpha
        cntrl = mdl.alpha
    elif params.inversion.beta_active:
        delta = params.inversion.delta_beta
        gamma = params.inversion.gamma_beta
        cntrl = mdl.beta

    reg_op = prior.laplacian(delta, gamma, cntrl.function_space())

    space = cntrl.function_space()
    x, y, z = [Function(space) for i in range(3)]


    # TODO: not convinced this does anything at present
    # Was used to test that eigenvectors are prior inverse orthogonal
    # test, trial = TestFunction(space), TrialFunction(space)
    # mass = assemble(inner(test,trial)*dx)
    # mass_solver = KrylovSolver("cg", "sor")
    # mass_solver.parameters.update({"absolute_tolerance":1.0e-32,
    #                            "relative_tolerance":1.0e-14})
    # mass_solver.set_operator(mass)


    # Loads eigenvalues from file
    with open(os.path.join(outdir, lamfile), 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)

    # and eigenvectors from .h5 file
    eps = params.constants.float_eps
    W = np.zeros((x.vector().size(),nlam))

    with HDF5File(MPI.comm_world, os.path.join(outdir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            hdf5data.read(x, f'v/vector_{i}')
            v = x.vector().get_local()
            reg_op.action(x.vector(), y.vector())

            tmp = y.vector().get_local()
            norm_in_prior = np.sqrt(np.dot(v,tmp))
            assert (abs(norm_in_prior - 1.0) < eps)
            W[:,i] = v



    # Take only the largest eigenvalues
    pind = np.flatnonzero(lam>threshlam)
    lam = lam[pind]
    W = W[:,pind]
    D = np.diag(1 / np.sqrt(1+lam) - 1)


########################################################3    
    
    if (os.path.exists(os.path.join(outdir,'step_number.npy'))):
        min_step = np.load(os.path.join(outdir,'step_number.npy'))
    else:
        min_step = 0


    if (os.path.exists(os.path.join(outdir,'sampling_results.npy'))):
        Qarray = np.load(os.path.join(outdir,'sampling_results.npy'))
    else:
        Qarray = np.zeros((params.time.num_sens,params.time.num_samples))
        min_step = 0

    slvr = solver.ssa_solver(mdl)

    
    for i in range(min_step,params.time.num_samples):
        info("sample number {0}".format(i))
        x, y, z, a = [Function(space) for i in range(4)]
        shp = np.shape(z.vector().get_local())
        np.random.seed()
        x.vector().set_local(random.normal(np.zeros(shp),  # N
                         np.ones(shp),shp))
        x.vector().apply("insert")
        
        reg_op.sqrt_action(x.vector(),y.vector())  # Gamma -1/2 N
        reg_op.sqrt_inv_action(x.vector(),z.vector())  # Gamma 1/2 N
        
        tmp1 = np.dot(W.T,y.vector().get_local())
        tmp2 = np.dot(D,tmp1)
        P1 = np.dot(W,tmp2)
        
        a.vector().set_local(z.vector().get_local() + P1)
        a.vector().apply("insert")

        """
        if(i==0):
         xpts    = mdl.mesh.coordinates()[:,0]
         ypts    = mdl.mesh.coordinates()[:,1]
         t    = mdl.mesh.cells()
         fig = plt.figure(figsize=(10,5))
         ax = fig.add_subplot(1,2,1)
         v    = z.compute_vertex_values(mdl.mesh)
         minv = np.min(v)
         maxv = np.max(v)
         levels = np.linspace(minv,maxv,20)
         ticks = np.linspace(minv,maxv,3)
         tick_options = {'axis':'both','which':'both','bottom':False,
                      'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}
         ax.tick_params(**tick_options)
         ax.text(0.05, 0.95, 'a', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
         c = ax.tricontourf(xpts, ypts, t, v, levels = levels, cmap='bwr')
         cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal")
         plt.tight_layout(2.0)
    
         ax = fig.add_subplot(1,2,2)
         v    = a.compute_vertex_values(mdl.mesh)
         minv = np.min(v)
         maxv = np.max(v)
         levels = np.linspace(minv,maxv,20)
         ticks = np.linspace(minv,maxv,3)
         tick_options = {'axis':'both','which':'both','bottom':False,
                     'top':False,'left':False,'right':False,'labelleft':False, 'labelbottom':False}
         ax.tick_params(**tick_options)
         ax.text(0.05, 0.95, 'b', transform=ax.transAxes,
             fontsize=13, fontweight='bold', va='top')
         c = ax.tricontourf(xpts, ypts, t, v, levels = levels, cmap='bwr')
         cbar = plt.colorbar(c, ticks=ticks, pad=0.05, orientation="horizontal")
         plt.tight_layout(2.0)
         plt.savefig(os.path.join(plotdir, 'sample.pdf'))
         plt.show()
        """

        a.vector()[:] += mdl.alpha.vector()[:]
        z.vector()[:] += mdl.alpha.vector()[:]
        slvr.alpha=a
        slvr.save_ts_zero()

        try:
            Q = slvr.timestep(save=0,adjoint_flag=0,cost_flag=1,qoi_func=slvr.comp_Q_h2 )
            for j in range(params.time.num_sens):
                Qarray[j,i] = Q[j].value()
        except: 
            info("something went wrong in solver")
            for j in range(params.time.num_sens):
                Qarray[j,i] = -9999.0

        slvr.reset_ts_zero()
        if(np.mod(i,5)==0):    
         np.save(os.path.join(outdir,'sampling_results'),Qarray)
         np.save(os.path.join(outdir,'step_number'),i)
         info("progress saved, step {0}".format(i))

    

if __name__ == "__main__":
    stop_annotating()

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_sample_post(sys.argv[1])
