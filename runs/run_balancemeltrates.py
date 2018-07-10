import sys
import os
import getopt
import argparse
sys.path.insert(0,'../code/')
from fenics import *
from dolfin import *
import model
import solver
import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
import time
import datetime
import pickle
from IPython import embed

def main(dd, outdir, run_length, n_steps, init_yr):

    #Load Data
    param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )

    param['outdir'] = outdir
    param['picard_params'] = {"nonlinear_solver":"newton",
                "newton_solver":{"linear_solver":"umfpack",
                "maximum_iterations":25,
                "absolute_tolerance":1.0e-3,
                "relative_tolerance":5.0e-2,
                "convergence_criterion":"incremental",
                "error_on_nonconvergence":False,
                "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}

    #Load Data
    mesh = Mesh(os.path.join(dd,'mesh.xml'))

    M = FunctionSpace(mesh, 'DG', 0)
    Q = FunctionSpace(mesh, 'Lagrange', 1) if os.path.isfile(os.path.join(dd,'param.p')) else M

    mask = Function(M,os.path.join(dd,'mask.xml'))

    if os.path.isfile(os.path.join(dd,'data_mesh.xml')):
        data_mesh = Mesh(os.path.join(dd,'data_mesh.xml'))
        Mdata = FunctionSpace(data_mesh, 'DG', 0)
        data_mask = Function(Mdata, os.path.join(dd,'data_mask.xml'))
    else:
        data_mesh = mesh
        data_mask = mask

    if not param['periodic_bc']:
       Qp = Q
       V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
    else:
       Qp = FunctionSpace(mesh,'Lagrange',1,constrained_domain=model.PeriodicBoundary(param['periodic_bc']))
       V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2,constrained_domain=model.PeriodicBoundary(param['periodic_bc']))


    #Load fields
    U = Function(V,os.path.join(dd,'U.xml'))

    alpha = Function(Qp,os.path.join(dd,'alpha.xml'))
    beta = Function(Qp,os.path.join(dd,'beta.xml'))
    bed = Function(Q,os.path.join(dd,'bed.xml'))

    thick = Function(M,os.path.join(dd,'thick.xml'))
    mask_vel = Function(M,os.path.join(dd,'mask_vel.xml'))
    u_obs = Function(M,os.path.join(dd,'u_obs.xml'))
    v_obs = Function(M,os.path.join(dd,'v_obs.xml'))
    u_std = Function(M,os.path.join(dd,'u_std.xml'))
    v_std = Function(M,os.path.join(dd,'v_std.xml'))
    uv_obs = Function(M,os.path.join(dd,'uv_obs.xml'))

    param['run_length'] =  run_length
    param['n_steps'] = n_steps

    mdl = model.model(data_mesh,data_mask, param)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
    mdl.init_lat_dirichletbc()
    mdl.init_bmelt(Constant(0.0))
    mdl.init_alpha(alpha)
    mdl.init_beta(beta)
    mdl.label_domain()

    #Solve
    slvr = solver.ssa_solver(mdl)
    slvr.save_ts_zero()
    slvr.timestep(save = 1, adjoint_flag=0)

    #Balance melt rates

    #Load time series of ice thicknesses
    hdf = HDF5File(slvr.mesh.mpi_comm(), param['outdir'] + 'H_ts.h5', "r")
    attr = hdf.attributes("H")
    nsteps = attr['count']

    #model time step
    dt= param['run_length']/param['n_steps']

    #Model iterations to difference between
    iter_s = np.ceil(init_yr/dt)  #Iteration closest to 5yr
    iter_f = nsteps - 1         #Final iteration
    dT = dt*(iter_f - iter_s) #Time diff in years between iterations

    #Read iteration data
    HS = Function(slvr.M)
    HF = Function(slvr.M)
    hdf.read(HS, "H/vector_{0}".format(int(iter_s)))
    hdf.read(HF, "H/vector_{0}".format(int(iter_f)))

    #Mask out grounded region
    H_s = -param['rhow']/param['rhoi'] * bed
    fl_ex = conditional(slvr.H_init <= H_s, Constant(1.0), Constant(0.0))

    #Calculate bmelt
    bmelt = project(max(fl_ex*(HF - HS)/dT, Constant(0.0)), slvr.M)

    #Output model variables in ParaView+Fenics friendly format
    outdir = mdl.param['outdir']
    pickle.dump( mdl.param, open( os.path.join(outdir,'param.p'), "wb" ) )

    File(os.path.join(outdir,'mesh.xml')) << mdl.mesh

    vtkfile = File(os.path.join(outdir,'bmelt.pvd'))
    xmlfile = File(os.path.join(outdir,'bmelt.xml'))
    vtkfile << bmelt
    xmlfile << bmelt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runlength', dest='run_length', type=float, help='Length of forward run in years (Default 10yrs)')
    parser.add_argument('-n', '--nsteps', dest='n_steps', type=int, help='Number of model timesteps (Default 240)')
    parser.add_argument('-y', '--yearinitial', dest='init_yr', type=int, help='The initial year to difference final model results with to calculate balance melt rates (Default 5yrs)')

    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')

    parser.set_defaults(run_length=10.0, n_steps=240, init_yr=5, outdir=False)
    args = parser.parse_args()

    run_length = args.run_length
    n_steps = args.n_steps
    init_yr = args.init_yr
    outdir = args.outdir
    dd = args.dd


    if init_yr >= run_length:
        print('Init year must less than the run length')
        sys.exit(2)

    if not outdir:
        outdir = ''.join(['./balance_melt_rates_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)


    if init_yr >= run_length:
        print('Init year must less than the run length')
        sys.exit(2)

    main(dd, outdir, run_length, n_steps, init_yr)
