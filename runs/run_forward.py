import sys
import os

import argparse
from fenics import *
from tlm_adjoint_fenics import *
import model
import solver
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
import time
import datetime
import pickle
from IPython import embed

stop_annotating()
np.random.seed(10)

def run_forward(config_file):

   # Determine Mesh

    # Create a new mesh with specific resolution
    if nx and ny:
        data_mesh_file = 'data_mesh.xml'
        data_mask_file = 'data_mask.xml'

        assert(os.path.isfile(os.path.join(dd,data_mesh_file))), 'Need data_mesh.xml to interpolate'
        assert(os.path.isfile(os.path.join(dd,data_mask_file))), 'Need data_mask.xml to interpolate'

        #Generate model mesh
        print('Generating new mesh')
        gf = 'grid_data.npz'
        npzfile = np.load(os.path.join(dd,'grid_data.npz'))
        xlim = npzfile['xlim']
        ylim = npzfile['ylim']

        mesh = RectangleMesh(Point(xlim[0],ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny)

    # Reuse a mesh; in this case, mesh and data_mesh will be identical

    # Otherwise see if there is previous run
    elif os.path.isfile(os.path.join(dd,'mesh.xml')):
        data_mesh_file = 'mesh.xml'
        data_mask_file = 'mask.xml'

        mesh = Mesh(os.path.join(dd,data_mesh_file))
    
    # Mirror data files
    elif os.path.isfile(os.path.join(dd,'data_mesh.xml')):
        #Start from raw data
        data_mesh_file = 'data_mesh.xml'
        data_mask_file = 'data_mask.xml'

        mesh = Mesh(os.path.join(dd,data_mesh_file))

    else:
        print('Need mesh and mask files')
        raise SystemExit

    data_mesh = Mesh(os.path.join(dd,data_mesh_file))
    
    # Define Function Spaces
    M = FunctionSpace(data_mesh, 'DG', 0)
    Q = FunctionSpace(data_mesh, 'Lagrange', 1)
    V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
    Qp = Q

    # Make necessary modification for periodic bc
    if periodic_bc:

        #If we're on a new mesh
        if nx and ny:
            L1 = xlim[-1] - xlim[0]
            L2 = ylim[-1] - ylim[0]
            assert( L1==L2), 'Periodic Boundary Conditions require a square domain'
            mesh_length = L1

        #If previous run   
        elif os.path.isfile(os.path.join(dd,'param.p')):
            mesh_length = pickle.load(open(os.path.join(dd,'param.p'), 'rb'))['periodic_bc']
            assert(mesh_length), 'Need to run periodic bc using original files'

        # Assume we're on a data_mesh
        else:
            gf = 'grid_data.npz'
            npzfile = np.load(os.path.join(dd,'grid_data.npz'))
            xlim = npzfile['xlim']
            ylim = npzfile['ylim']
            L1 = xlim[-1] - xlim[0]
            L2 = ylim[-1] - ylim[0]
            assert( L1==L2), 'Periodic Boundary Conditions require a square domain'
            mesh_length = L1

        Qp = FunctionSpace(data_mesh,'Lagrange',1,constrained_domain=model.PeriodicBoundary(mesh_length))
        V = VectorFunctionSpace(data_mesh,'Lagrange',1,dim=2,constrained_domain=model.PeriodicBoundary(mesh_length))
    

    data_mask = Function(M,os.path.join(dd,data_mask_file))


    #Load fields
    U = Function(V,os.path.join(dd,'U.xml'))

    alpha = Function(Qp,os.path.join(dd,'alpha.xml'))
    beta = Function(Qp,os.path.join(dd,'beta.xml'))
    bed = Function(Q,os.path.join(dd,'bed.xml'))

    bmelt = Function(M,os.path.join(dd,'bmelt.xml'))
    smb = Function(M,os.path.join(dd,'smb.xml'))
    thick = Function(M,os.path.join(dd,'thick.xml'))
    mask = Function(M,os.path.join(dd,'mask.xml'))
    mask_vel = Function(M,os.path.join(dd,'mask_vel.xml'))
    u_obs = Function(M,os.path.join(dd,'u_obs.xml'))
    v_obs = Function(M,os.path.join(dd,'v_obs.xml'))
    u_std = Function(M,os.path.join(dd,'u_std.xml'))
    v_std = Function(M,os.path.join(dd,'v_std.xml'))
    uv_obs = Function(M,os.path.join(dd,'uv_obs.xml'))


    #Load Data
    param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )
    param['sliding_law'] = sl

    param['outdir'] = outdir
    if sl == 0:
        param['picard_params'] =  {"nonlinear_solver":"newton",
                                "newton_solver":{"linear_solver":"umfpack",
                                "maximum_iterations":200,
                                "absolute_tolerance":1.0e-0,
                                "relative_tolerance":1.0e-3,
                                "convergence_criterion":"incremental",
                                "error_on_nonconvergence":False,
                                }}
        param['newton_params'] =  {"nonlinear_solver":"newton",
                                "newton_solver":{"linear_solver":"umfpack",
                                "maximum_iterations":25,
                                "absolute_tolerance":1.0e-7,
                                "relative_tolerance":1.0e-8,
                                "convergence_criterion":"incremental",
                                "error_on_nonconvergence":True,
                                }}

    elif sl == 1:
        param['picard_params'] =  {"nonlinear_solver":"newton",
                                "newton_solver":{"linear_solver":"umfpack",
                                "maximum_iterations":200,
                                "absolute_tolerance":1.0e-4,
                                "relative_tolerance":1.0e-10,
                                "convergence_criterion":"incremental",
                                "error_on_nonconvergence":False,
                                }}
        param['newton_params'] =  {"nonlinear_solver":"newton",
                                "newton_solver":{"linear_solver":"umfpack",
                                "maximum_iterations":25,
                                "absolute_tolerance":1.0e-4,
                                "relative_tolerance":1.0e-5,
                                "convergence_criterion":"incremental",
                                "error_on_nonconvergence":True,
                                }}


    param['run_length'] =  run_length
    param['n_steps'] = n_steps
    param['num_sens'] = num_sens

    mdl = model.model(mesh,data_mask, param)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
    mdl.init_lat_dirichletbc()
    mdl.init_bmelt(bmelt)
    mdl.init_smb(smb)
    mdl.init_alpha(alpha)
    mdl.init_beta(beta) #TODO <- should this be perturbed? likewise in other run_*.py
    mdl.label_domain()

    #Solve
    slvr = solver.ssa_solver(mdl)
    slvr.save_ts_zero()

    opts = {'0': slvr.alpha, '1': slvr.beta, '2': [slvr.alpha,slvr.beta]}
    cntrl = opts[str(pflag)]

    qoi_func =  slvr.comp_Q_h2 if qoi == 1 else slvr.comp_Q_vaf
    Q = slvr.timestep(adjoint_flag=1, qoi_func=qoi_func)
    dQ_ts = compute_gradient(Q, cntrl) #Isaac 27

    #Uncomment for Taylor Verification, Comment above two lines
    # param['num_sens'] = 1
    # J = slvr.timestep(adjoint_flag=1, cst_func=slvr.comp_Q_vaf)
    # dJ = compute_gradient(J, slvr.alpha)
    #
    #
    # def forward_ts(alpha_val=None):
    #     slvr.reset_ts_zero()
    #     if alpha_val:
    #         slvr.alpha = alpha_val
    #     return slvr.timestep(adjoint_flag=1, cst_func=slvr.comp_Q_vaf)
    #
    #
    # min_order = taylor_test(lambda alpha : forward_ts(alpha_val = alpha), slvr.alpha,
    #   J_val = J.value(), dJ = dJ, seed = 1e-2, size = 6)
    # sys.exit(os.EX_OK)

    #Output model variables in ParaView+Fenics friendly format
    outdir = mdl.param['outdir']
    pickle.dump( mdl.param, open( os.path.join(outdir,'param.p'), "wb" ) )

    File(os.path.join(outdir,'mesh.xml')) << mdl.mesh

    ts = np.linspace(0,run_length,n_steps+1)
    pickle.dump([slvr.Qval_ts, ts], open( os.path.join(outdir,'Qval_ts.p'), "wb" ) )


    vtkfile = File(os.path.join(outdir,'dQ_ts.pvd'))
    hdf5out = HDF5File(MPI.comm_world, os.path.join(outdir, 'dQ_ts.h5'), 'w')
    n=0.0

    for j in dQ_ts:
        j.rename('dQ', 'dQ')
        vtkfile << j
        hdf5out.write(j, 'dQ', n)
        n += 1.0

    hdf5out.close()


    vtkfile = File(os.path.join(outdir,'U.pvd'))
    xmlfile = File(os.path.join(outdir,'U.xml'))
    vtkfile << slvr.U
    xmlfile << slvr.U

    vtkfile = File(os.path.join(outdir,'beta.pvd'))
    xmlfile = File(os.path.join(outdir,'beta.xml'))
    vtkfile << slvr.beta
    xmlfile << slvr.beta

    vtkfile = File(os.path.join(outdir,'bed.pvd'))
    xmlfile = File(os.path.join(outdir,'bed.xml'))
    vtkfile << mdl.bed
    xmlfile << mdl.bed

    vtkfile = File(os.path.join(outdir,'thick.pvd'))
    xmlfile = File(os.path.join(outdir,'thick.xml'))
    H = project(mdl.H, mdl.M)
    vtkfile << H
    xmlfile << H

    vtkfile = File(os.path.join(outdir,'mask.pvd'))
    xmlfile = File(os.path.join(outdir,'mask.xml'))
    vtkfile << mdl.mask
    xmlfile << mdl.mask

    vtkfile = File(os.path.join(outdir,'mask_vel.pvd'))
    xmlfile = File(os.path.join(outdir,'mask_vel.xml'))
    vtkfile << mdl.mask_vel
    xmlfile << mdl.mask_vel

    vtkfile = File(os.path.join(outdir,'u_obs.pvd'))
    xmlfile = File(os.path.join(outdir,'u_obs.xml'))
    vtkfile << mdl.u_obs
    xmlfile << mdl.u_obs

    vtkfile = File(os.path.join(outdir,'v_obs.pvd'))
    xmlfile = File(os.path.join(outdir,'v_obs.xml'))
    vtkfile << mdl.v_obs
    xmlfile << mdl.v_obs

    vtkfile = File(os.path.join(outdir,'u_std.pvd'))
    xmlfile = File(os.path.join(outdir,'u_std.xml'))
    vtkfile << mdl.u_std
    xmlfile << mdl.u_std

    vtkfile = File(os.path.join(outdir,'v_std.pvd'))
    xmlfile = File(os.path.join(outdir,'v_std.xml'))
    vtkfile << mdl.v_std
    xmlfile << mdl.v_std

    vtkfile = File(os.path.join(outdir,'uv_obs.pvd'))
    xmlfile = File(os.path.join(outdir,'uv_obs.xml'))
    U_obs = project((mdl.v_obs**2 + mdl.u_obs**2)**(1.0/2.0), mdl.M)
    vtkfile << U_obs
    xmlfile << U_obs

    vtkfile = File(os.path.join(outdir,'alpha.pvd'))
    xmlfile = File(os.path.join(outdir,'alpha.xml'))
    vtkfile << slvr.alpha
    xmlfile << slvr.alpha


    vtkfile = File(os.path.join(outdir,'surf.pvd'))
    xmlfile = File(os.path.join(outdir,'surf.xml'))
    vtkfile << mdl.surf
    xmlfile << mdl.surf


if __name__ == "__main__":
    stop_annotating()

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', dest='run_length', type=float, required=True, help='Number of years to run for')
    parser.add_argument('-n', '--num_timesteps', dest='n_steps', type=int, required=True, help='Number of timesteps')
    parser.add_argument('-b', '--boundaries', dest='periodic_bc', action='store_true', help='Periodic boundary conditions')
    parser.add_argument('-x', '--cells_x', dest='nx', type=int, help='Number of cells in x direction')
    parser.add_argument('-y', '--cells_y', dest='ny', type=int, help='Number of cells in y direction')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')
    parser.add_argument('-s', '--num_sens', dest='num_sens', type=int, help='Number of samples of cost function')
    parser.add_argument('-p', '--parameters', dest='pflag', choices=[0, 1, 2], type=int, help='Parameter to calculate sensitivity to: alpha (0), beta (1), [Future->] alpha and beta (2)')
    parser.add_argument('-q', '--slidinglaw', dest='sl', type=float,  help = 'Sliding Law (0: linear (default), 1: weertman)')
    parser.add_argument('-i', '--quantity_of_interest', dest='qoi', type=float,  help = 'Quantity of interest (0: VAF (default), 1: H^2 (for ISMIPC))')

    parser.set_defaults(periodic_bc = False, nx=False,ny=False, outdir=False, num_sens = 1.0, pflag=0,sl=0, qoi=0)
    args = parser.parse_args()

    n_steps = args.n_steps
    run_length = args.run_length
    periodic_bc = args.periodic_bc
    outdir = args.outdir
    dd = args.dd
    nx = args.nx
    ny = args.ny
    num_sens = args.num_sens
    pflag = args.pflag
    sl = args.sl
    qoi = args.qoi

    if not outdir:
        outdir = ''.join(['./run_forward_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_forward(sys.argv[1])
