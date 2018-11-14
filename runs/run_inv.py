import sys
sys.path.insert(0,'../code/')
sys.path.insert(0,'../../dolfin_adjoint_custom/python/')

import os
import argparse
from dolfin import *
from tlm_adjoint import *

import model
import solver
import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
import time
import datetime
import pickle
from IPython import embed


def main(maxiter, rc_inv, pflag, outdir, dd, nx, ny, sim_flag, bflag, altiter, sl):

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


    bed = Function(Q,os.path.join(dd,'bed.xml'))

    thick = Function(M,os.path.join(dd,'thick.xml'))
    u_obs = Function(M,os.path.join(dd,'u_obs.xml'))
    v_obs = Function(M,os.path.join(dd,'v_obs.xml'))
    u_std = Function(M,os.path.join(dd,'u_std.xml'))
    v_std = Function(M,os.path.join(dd,'v_std.xml'))
    mask_vel = Function(M,os.path.join(dd,'mask_vel.xml'))
    Bglen = Function(M,os.path.join(dd,'Bglen.xml'))
    bmelt = Function(M,os.path.join(dd,'bmelt.xml'))
    smb = Function(M,os.path.join(dd,'smb.xml'))

    if not os.path.isfile(os.path.join(dd,'param.p')):
        print('Generating new mesh')
        #Generate model mesh
        gf = 'grid_data.npz'
        npzfile = np.load(os.path.join(dd,'grid_data.npz'))
        xlim = npzfile['xlim']
        ylim = npzfile['ylim']

        if not nx:
            nx = int(npzfile['nx'])
        if not ny:
            ny = int(npzfile['ny'])

        mesh = RectangleMesh(Point(xlim[0],ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny)
    else:
        print('Identified as previous run, reusing mesh')


    if bflag:
        if os.path.isfile(os.path.join(dd,'param.p')):
            bflag = pickle.load(open(os.path.join(dd,'param.p'), 'rb'))['periodic_bc']
            assert(bflag), 'Need to run periodic bc using original files'
        else:
            L1 = xlim[-1] - xlim[0]
            L2 = ylim[-1] - ylim[0]
            assert( L1==L2), 'Periodic Boundary Conditions require a square domain'
            bflag = L1





    #Initialize Model
    param = {
            'outdir' : outdir,
            'rc_inv': rc_inv,
            'pflag': pflag,
            'sim_flag': sim_flag,
            'periodic_bc': bflag,
            'altiter': altiter,
            'sliding_law': sl,
            'inv_options': {'maxiter': maxiter, 'disp': True, 'ftol': 1e-4}}

    if sl == 0:
        param['picard_params'] =  {"nonlinear_solver":"newton",
                                "newton_solver":{"linear_solver":"umfpack",
                                "maximum_iterations":200,
                                "absolute_tolerance":1.0e-0,
                                "relative_tolerance":1.0e-3,
                                "convergence_criterion":"incremental",
                                "error_on_nonconvergence":False,
                                "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}
        param['newton_params'] =  {"nonlinear_solver":"newton",
                                "newton_solver":{"linear_solver":"umfpack",
                                "maximum_iterations":25,
                                "absolute_tolerance":1.0e-7,
                                "relative_tolerance":1.0e-8,
                                "convergence_criterion":"incremental",
                                "error_on_nonconvergence":True,
                                "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}

    elif sl == 1:
        param['picard_params'] =  {"nonlinear_solver":"newton",
                                "newton_solver":{"linear_solver":"umfpack",
                                "maximum_iterations":200,
                                "absolute_tolerance":1.0e-4,
                                "relative_tolerance":1.0e-10,
                                "convergence_criterion":"incremental",
                                "error_on_nonconvergence":False,
                                "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}
        param['newton_params'] =  {"nonlinear_solver":"newton",
                                "newton_solver":{"linear_solver":"umfpack",
                                "maximum_iterations":25,
                                "absolute_tolerance":1.0e-4,
                                "relative_tolerance":1.0e-5,
                                "convergence_criterion":"incremental",
                                "error_on_nonconvergence":True,
                                "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}


    mdl = model.model(mesh,data_mask, param)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
    mdl.init_lat_dirichletbc()
    mdl.init_bmelt(bmelt)
    mdl.init_smb(smb)
    mdl.label_domain()

    mdl.gen_alpha()
    mdl.init_beta(mdl.bglen_to_beta(Bglen))            #Comment to use uniform Bglen

    #Inversion
    slvr = solver.ssa_solver(mdl)

    opts = {'0': [slvr.alpha], '1': [slvr.beta], '2': [slvr.alpha,slvr.beta]}
    slvr.inversion(opts[str(pflag)])


    #Output model variables in ParaView+Fenics friendly format
    outdir = mdl.param['outdir']
    pickle.dump( mdl.param, open( os.path.join(outdir,'param.p'), "wb" ) )

    File(os.path.join(outdir,'mesh.xml')) << mdl.mesh
    File(os.path.join(outdir,'data_mesh.xml')) << data_mesh


    vtkfile = File(os.path.join(outdir,'U.pvd'))
    xmlfile = File(os.path.join(outdir,'U.xml'))
    vtkfile << slvr.U
    xmlfile << slvr.U

    vtkfile = File(os.path.join(outdir,'beta.pvd'))
    xmlfile = File(os.path.join(outdir,'beta.xml'))
    vtkfile << slvr.beta
    xmlfile << slvr.beta

    vtkfile = File(os.path.join(outdir,'beta_bgd.pvd'))
    xmlfile = File(os.path.join(outdir,'beta_bgd.xml'))
    vtkfile << slvr.beta_bgd
    xmlfile << slvr.beta_bgd

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

    vtkfile = File(os.path.join(outdir,'data_mask.pvd'))
    xmlfile = File(os.path.join(outdir,'data_mask.xml'))
    vtkfile << mask
    xmlfile << mask

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

    vtkfile = File(os.path.join(outdir,'Bglen.pvd'))
    xmlfile = File(os.path.join(outdir,'Bglen.xml'))
    Bglen = project(slvr.beta_to_bglen(slvr.beta),mdl.M)
    vtkfile << Bglen
    xmlfile << Bglen

    vtkfile = File(os.path.join(outdir,'bmelt.pvd'))
    xmlfile = File(os.path.join(outdir,'bmelt.xml'))
    vtkfile << slvr.bmelt
    xmlfile << slvr.bmelt

    vtkfile = File(os.path.join(outdir,'smb.pvd'))
    xmlfile = File(os.path.join(outdir,'smb.xml'))
    vtkfile << slvr.smb
    xmlfile << slvr.smb

    vtkfile = File(os.path.join(outdir,'surf.pvd'))
    xmlfile = File(os.path.join(outdir,'surf.xml'))
    vtkfile << mdl.surf
    xmlfile << mdl.surf

if __name__ == "__main__":
    stop_annotating()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--maxiter', dest='maxiter', type=int, help='Maximum number of inversion iterations')
    parser.add_argument('-r', '--rc_inv', dest='rc_inv', nargs=5, type=float, required=True, help='Scaling Constants')
    parser.add_argument('-p', '--parameters', dest='pflag', choices=[0, 1, 2], type=int, required=True, help='Inversion parameters: alpha (0), beta (1), alpha and beta (2)')
    parser.add_argument('-s', '--simultaneousmethod', dest='sim_flag', action='store_true', help='Dual parameter inversion for both parameters simultaneously (default is to alternative through parameters)')
    parser.add_argument('-a', '--altiter', dest='altiter', type=int, help='Number of times to iterate through parameters for inversions w/ more than one parameter (not applicable when conducting dual inversion)')
    parser.add_argument('-x', '--cells_x', dest='nx', type=int, help='Number of cells in x direction (defaults to data resolution)')
    parser.add_argument('-y', '--cells_y', dest='ny', type=int, help='Number of cells in y direction (defaults to data resolution)')
    parser.add_argument('-b', '--boundaries', dest='bflag', action='store_true', help='Periodic boundary conditions')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')
    parser.add_argument('-q', '--slidinglaw', dest='sl', type=float,  help = 'Sliding Law (0: linear (default), 1: weertman)')

    parser.set_defaults(maxiter=15,nx=False,ny=False,sim_flag=False, bflag = False, altiter=2, sl=0)
    args = parser.parse_args()

    maxiter = args.maxiter
    rc_inv = args.rc_inv
    pflag = args.pflag
    outdir = args.outdir
    dd = args.dd
    nx = args.nx
    ny = args.ny
    sim_flag = args.sim_flag
    bflag = args.bflag
    altiter = args.altiter
    sl = args.sl

    if not outdir:
        outdir = ''.join(['./run_inv_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)



    main(maxiter, rc_inv, pflag, outdir, dd, nx, ny, sim_flag, bflag, altiter, sl)
