#!/usr/bin/env python
import sys
sys.path.insert(0,'../../dolfin_adjoint_custom/python/')
sys.path.insert(0,'../code/')

import os
import argparse
from fenics import *
from tlm_adjoint import *
import pickle
from IPython import embed
import model
import solver
import datetime
import numpy as np
from eigendecomposition_custom import *
import matplotlib.pyplot as plt
import prior


def main(num_eig, n_iter, slepsc_flag, msft_flag, pflag, gnhep, outdir, dd, fileout):

    #Load parameters of run
    param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )
    rc_inv = param['rc_inv']

    if msft_flag:
        #Set delta, gamma to << 1 (not zero!) in param[];
        #rc_inv contains original values for computing preconditioner
        tmp = list(rc_inv) #deepcopy
        tmp[1:] = [1e-30 for i in rc_inv[1:]]
        param['rc_inv'] = tmp

    #Ice only mesh
    mesh = Mesh(os.path.join(dd,'mesh.xml'))

    #Set up Function spaces
    Q = FunctionSpace(mesh,'Lagrange',1)
    M = FunctionSpace(mesh,'DG',0)

    #Handle function with optional periodic boundary conditions
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
    mask = Function(M,os.path.join(dd,'mask.xml'))
    mask_vel = Function(M,os.path.join(dd,'mask_vel.xml'))
    u_obs = Function(M,os.path.join(dd,'u_obs.xml'))
    v_obs = Function(M,os.path.join(dd,'v_obs.xml'))
    u_std = Function(M,os.path.join(dd,'u_std.xml'))
    v_std = Function(M,os.path.join(dd,'v_std.xml'))
    uv_obs = Function(M,os.path.join(dd,'uv_obs.xml'))
    Bglen = Function(M,os.path.join(dd,'Bglen.xml'))


    #Initialize our model object
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

    #Setup our solver object
    slvr = solver.ssa_solver(mdl)

    opts = {'0': slvr.alpha, '1': slvr.beta, '2': [slvr.alpha,slvr.beta]}
    cntrl = opts[str(pflag)]
    space = cntrl.function_space()


    #Hessian Action
    slvr.set_hessian_action(cntrl)

    #Mass matrix solver
    xg,xb = Function(space), Function(space)
    test, trial = TestFunction(space), TrialFunction(space)
    mass = assemble(inner(test,trial)*slvr.dx)
    mass_solver = KrylovSolver("cg", "sor")
    mass_solver.parameters.update({"absolute_tolerance":1.0e-32,
                               "relative_tolerance":1.0e-14})
    mass_solver.set_operator(mass)

    #Regularization operator using inversion delta/gamma values

    if pflag == 0:
        delta = rc_inv[1]
        gamma = rc_inv[3]
    elif pflag == 1:
        delta = rc_inv[2]
        gamma = rc_inv[4]

    reg_op = prior.laplacian(delta,gamma, space)

    #Counter for hessian action -- list rather than float/int necessary
    num_action_calls = [0]

    def gnhep_prior_action(x):
        num_action_calls[0] += 1
        info("gnhep_prior_action call %i" % num_action_calls[0])
        _, _, ddJ_val = slvr.ddJ.action(cntrl, x)
        reg_op.inv_action(ddJ_val.vector(), xg.vector())
        return function_get_values(xg)

    def gnhep_mass_action(x):
        num_action_calls[0] += 1
        info("gnhep_mass_action call %i" % num_action_calls[0])
        _, _, ddJ_val = slvr.ddJ.action(cntrl, x)
        mass_solver.solve(xg.vector(), ddJ_val.vector())
        return function_get_values(xg)


    opts = {'0': gnhep_prior_action, '1': gnhep_mass_action}
    gnhep_func = opts[str(gnhep)]

    #Hessian eigendecomposition using SLEPSC
    if slepsc_flag:

        #Eigendecomposition
        lam, [vr, vi] = eigendecompose(space, gnhep_func, tolerance = 1.0e-10, N_eigenvalues = num_eig)

        # Uses extreme amounts of disk space; suitable for ismipc only
        # #Save eigenfunctions
        # vtkfile = File(os.path.join(outdir,'vr.pvd'))
        # for v in vr:
        #     v.rename('v', v.label())
        #     vtkfile << v
        #
        # vtkfile = File(os.path.join(outdir,'vi.pvd'))
        # for v in vi:
        #     v.rename('v', v.label())
        #     vtkfile << v

        hdf5file = HDF5File(slvr.mesh.mpi_comm(), os.path.join(outdir, 'vr.h5'), 'w')
        for i, v in enumerate(vr): hdf5file.write(v, 'v', i)

        hdf5file = HDF5File(slvr.mesh.mpi_comm(), os.path.join(outdir, 'vi.h5'), 'w')
        for i, v in enumerate(vi): hdf5file.write(v, 'v', i)



    #Save eigenvals and some associated info
    pfile = open( os.path.join(outdir, fileout), "wb" )
    pickle.dump( [lam, num_eig, n_iter, slepsc_flag, msft_flag, outdir, dd], pfile)
    pfile.close()

    #Plot of eigenvals
    lamr = lam.real
    lpos = np.argwhere(lamr > 0)
    lneg = np.argwhere(lamr < 0)
    lind = np.arange(0,len(lamr))
    plt.semilogy(lind[lpos], lamr[lpos], '.')
    plt.semilogy(lind[lneg], np.abs(lamr[lneg]), '.')
    plt.savefig(os.path.join(outdir,'lambda.pdf'))


if __name__ == "__main__":
    stop_annotating()


    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--numeig', dest='num_eig', type=int, help='Number of eigenvalues to find (default is all)')
    parser.add_argument('-i', '--niter', dest='n_iter', type=int, help='Number of power iterations for random algorithm')
    parser.add_argument('-s', '--slepsc', dest='slepsc_flag', action='store_true', help='Use slepsc instead of random algorithm')
    parser.add_argument('-m', '--msft_flag', dest='msft_flag', action='store_true', help='Consider only the misfit term of the cost function without regularization')
    parser.add_argument('-p', '--parameters', dest='pflag', choices=[0, 1, 2], type=int, required=True, help='Inversion parameters: alpha (0), beta (1), alpha and beta (2)')
    parser.add_argument('-g', '--gnhep', dest='gnhep', choices=[0, 1], type=int, help='Eigenvalue problem to solve: 0: prior preconditioned hessian (default); 1: inverse mass matrix preconditioned hessian')

    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')
    parser.add_argument('-f', '--fileout', dest='fileout', type=str, help='File to store eigenvalues')

    parser.set_defaults(n_iter=1, num_eig = None, slepsc_flag=False, msft_flag=False, outdir=False, fileout=False, gnhep=0)
    args = parser.parse_args()

    num_eig = args.num_eig
    n_iter = args.n_iter
    slepsc_flag = args.slepsc_flag
    msft_flag = args.msft_flag
    pflag = args.pflag
    gnhep = args.gnhep
    outdir = args.outdir
    dd = args.dd
    fileout = args.fileout

    if not outdir:
        outdir = ''.join(['./run_eigendec_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    if not fileout:
        if slepsc_flag: fileout = 'slepceig{0}{1}_{2}.p'.format(num_eig, 'm' if msft_flag else '', timestamp)
        else: fileout = 'randeig{0}{1}_{2}.p'.format(num_eig, 'm' if msft_flag else '', timestamp)

    main(num_eig, n_iter, slepsc_flag, msft_flag, pflag, gnhep, outdir, dd, fileout)
