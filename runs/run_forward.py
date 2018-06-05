import sys
import os
sys.path.insert(0,'../../dolfin_adjoint_custom/python/')
sys.path.insert(0,'../code/')

import argparse
from fenics import *
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

np.random.seed(10)

def main(n_steps,run_length,bflag, outdir, dd):

    #Load Data
    param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )

    param['outdir'] = outdir
    param['picard_params'] = {"nonlinear_solver":"newton",
                "newton_solver":{"linear_solver":"umfpack",
                "maximum_iterations":200,
                "absolute_tolerance":1.0e-8,
                "relative_tolerance":5.0e-3,
                "convergence_criterion":"incremental",
                "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}






    mesh = Mesh(os.path.join(dd,'mesh.xml'))

    V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
    Q = FunctionSpace(mesh,'Lagrange',1)
    M = FunctionSpace(mesh,'DG',0)

    #def forward(alpha_val = None):
    U = Function(V,os.path.join(dd,'U.xml'))
    alpha = Function(Q,os.path.join(dd,'alpha.xml'))
    beta = Function(Q,os.path.join(dd,'beta.xml'))
    bed = Function(Q,os.path.join(dd,'bed.xml'))
    surf = Function(Q,os.path.join(dd,'surf.xml'))
    thick = Function(M,os.path.join(dd,'thick.xml'))
    mask = Function(M,os.path.join(dd,'mask.xml'))
    mask_vel = Function(M,os.path.join(dd,'mask_vel.xml'))
    u_obs = Function(M,os.path.join(dd,'u_obs.xml'))
    v_obs = Function(M,os.path.join(dd,'v_obs.xml'))
    u_std = Function(M,os.path.join(dd,'u_std.xml'))
    v_std = Function(M,os.path.join(dd,'v_std.xml'))
    uv_obs = Function(M,os.path.join(dd,'uv_obs.xml'))



    param['run_length'] =  run_length
    param['n_steps'] = n_steps

    mdl = model.model(mesh,mask, param)
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
    slvr.comp_dJ_vaf(slvr.alpha)

    #J = slvr.timestep(adjoint_flag=1)
    #dJ = compute_gradient(J, slvr.alpha)

    # #Uncomment for Taylor Verification, Comment out slvr.comp_dJ_vaf
    # J = slvr.timestep(adjoint_flag=1)
    # dJ = compute_gradient(J, slvr.alpha)
    #
    # def forward_ts(alpha_val=None):
    #     slvr.reset_ts_zero()
    #     if alpha_val:
    #         slvr.alpha = alpha_val
    #     return slvr.timestep(adjoint_flag=1)
    #
    #
    # min_order = taylor_test(lambda alpha : forward_ts(alpha_val = alpha), slvr.alpha,
    #   J_val = J.value(), dJ = dJ, seed = 1e0, size = 6)
    # sys.exit(os.EX_OK)


    #Output model variables in ParaView+Fenics friendly format
    outdir = mdl.param['outdir']
    pickle.dump( mdl.param, open( os.path.join(outdir,'param.p'), "wb" ) )

    File(os.path.join(outdir,'mesh.xml')) << mdl.mesh

    vtkfile = File(os.path.join(outdir,'dJ_vaf.pvd'))
    xmlfile = File(os.path.join(outdir,'dJ_vaf.xml'))
    vtkfile << slvr.dJ_vaf
    xmlfile << slvr.dJ_vaf

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
    vtkfile << mdl.alpha
    xmlfile << mdl.alpha

    vtkfile = File(os.path.join(outdir,'Bglen.pvd'))
    xmlfile = File(os.path.join(outdir,'Bglen.xml'))
    Bglen = project(exp(mdl.beta),mdl.M)
    vtkfile << Bglen
    xmlfile << Bglen

    vtkfile = File(os.path.join(outdir,'surf.pvd'))
    xmlfile = File(os.path.join(outdir,'surf.xml'))
    vtkfile << mdl.surf
    xmlfile << mdl.surf


if __name__ == "__main__":
    stop_annotating()

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', dest='run_length', type=float, required=True, help='Number of years to run for')
    parser.add_argument('-n', '--num_timesteps', dest='n_steps', type=int, required=True, help='Number of timesteps')
    parser.add_argument('-b', '--boundaries', dest='bflag', action='store_true', help='Periodic boundary conditions')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')

    parser.set_defaults(bflag = False, outdir=False)
    args = parser.parse_args()

    n_steps = args.n_steps
    run_length = args.run_length
    bflag = args.bflag
    outdir = args.outdir
    dd = args.dd

    if not outdir:
        outdir = ''.join(['./run_forward_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    main(n_steps,run_length,bflag, outdir, dd)
