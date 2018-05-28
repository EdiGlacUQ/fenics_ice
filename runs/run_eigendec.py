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


def main(num_eig, n_iter, slepsc_flag, msft_flag, pflag, outdir, dd):

    #Load parameters of run
    param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )

    param['picard_params'] = {"nonlinear_solver":"newton",
                "newton_solver":{"linear_solver":"umfpack",
                "maximum_iterations":200,
                "absolute_tolerance":1.0e-8,
                "relative_tolerance":5.0e-3,
                "convergence_criterion":"incremental",
                "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}

    if msft_flag:
        tmp = param['rc_inv']
        tmp[1:] = [0 for i in tmp[1:]]
        param['rc_inv'] = tmp


    #Ice only mesh
    mesh = Mesh(os.path.join(dd,'mesh.xml'))

    #Set up Function spaces
    if not param['periodic_bc']:
       V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
    else:
       V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2,constrained_domain=model.PeriodicBoundary(self.param['periodic_bc']))

    Q = FunctionSpace(mesh,'Lagrange',1)
    M = FunctionSpace(mesh,'DG',0)

    #Load fields
    U = Function(V,os.path.join(dd,'U.xml'))

    alpha = Function(Q,os.path.join(dd,'alpha.xml'))
    beta = Function(Q,os.path.join(dd,'beta.xml'))
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

    opts = {'0': slvr.alpha, '1': [slvr.beta], '2': [slvr.alpha,slvr.beta]}
    cntrl = opts[str(pflag)]

    slvr.set_hessian_action(slvr.alpha)

    A_action =  slvr.ddJ.action_fn(cntrl)
    space = slvr.alpha.function_space()

    xg,xb = Function(space), Function(space)
    test, trial = TestFunction(space), TrialFunction(space)
    mass = assemble(inner(test,trial)*slvr.dx)
    mass_solver = KrylovSolver("cg", "sor")
    mass_solver.parameters.update({"absolute_tolerance":1.0e-32,
                               "relative_tolerance":1.0e-14})
    mass_solver.set_operator(mass)

    def gnhep_action(x):
        x = function_copy(x, static = True)
        _, _, ddJ_val = slvr.ddJ.action(cntrl, x)
        mass_solver.solve(xg.vector(), ddJ_val.vector())
        return function_get_values(xg)



    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")

    if slepsc_flag:
        lam, [vr, vi] = eigendecompose(space, gnhep_action, tolerance = 1.0e-2, N_eigenvalues = num_eig)
        fo = 'slepceig{0}{1}_{2}.p'.format(num_eig, 'm' if msft_flag else '', timestamp)

        vtkfile = File(os.path.join(outdir,'vr.pvd'))
        for v in vr:
            v.rename('v', v.label())
            vtkfile << v

        vtkfile = File(os.path.join(outdir,'vi.pvd'))
        for v in vi:
            v.rename('v', v.label())
            vtkfile << v

        hdf5file = HDF5File(slvr.mesh.mpi_comm(), os.path.join(outdir, 'vr.h5'), 'w')
        for i, v in enumerate(vr): hdf5file.write(v, 'v', i)

        hdf5file = HDF5File(slvr.mesh.mpi_comm(), os.path.join(outdir, 'vi.h5'), 'w')
        for i, v in enumerate(vi): hdf5file.write(v, 'v', i)



    else:
        lam,vv = randeig(space, A_action,k=num_eig,n_iter=n_iter)
        fo = 'randeig{0}{1}_{2}.p'.format(num_eig, 'm' if msft_flag else '', timestamp)



    pfile = open( os.path.join(outdir, fo), "wb" )
    pickle.dump( [lam, num_eig, n_iter, slepsc_flag, msft_flag, outdir, dd], pfile)
    pfile.close()

    plt.semilogy(lam.real)
    plt.savefig(os.path.join(outdir,'lambda.pdf'))

    print('Finished')


    #A = HessWrapper(A_action,space)
    #
    # #Sanity checks on eigenvalues/eigenvectors.
    # neg_ind = np.nonzero(lam<0)[0]
    # print('Number of eigenvalues: {0}'.format(num_eig))
    # print('Number of negative eigenvalues: {0}'.format(len(neg_ind)))
    #
    # cntr_nn = 0
    # cntr_np = 0
    #
    # print('Checking negative eigenvalues...')
    # #For each negative eigenvalue, independently recalculate their value using its eigenvector
    # for i in neg_ind:
    #     ev = vv[:,i]
    #     ll = lam[i]
    #     ll2 = np.dot(ev, A.apply(ev)) / np.dot(ev,ev)
    #
    #     #Compare signs of the calculated eigenvals, increment correct counter
    #     if np.sign(ll) == np.sign(ll2):
    #         cntr_nn += 1
    #     else:
    #         cntr_np +=1
    #         print('Eigenval {0} at index {1} is a false negative'.format(ll,i))
    #
    # print('The number of verified negative eigenvals is {0}'.format(cntr_nn))
    # print('The sign of {0} eigenvals is not supported by independent calculation using its eigenvector'.format(cntr_np))
    #
    #
    # print('Checking the accuracy of eigenval/eigenvec pairs...')
    #
    # npair = min(num_eig, 10) #Select a maximum of 10 pairs
    # f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)] #selector function
    # pind = f(npair,num_eig)
    #
    # for i in pind:
    #     ev = vv[:,i]
    #     ll = lam[i]
    #     res = np.linalg.norm(A.apply(ev) - ll*ev)/ll
    #     print('Residual of {0} at index {1}'.format(res, i))
    #
    #
    #
    #

if __name__ == "__main__":
    stop_annotating()


    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--numeig', dest='num_eig', type=int, required=True, help='Number of eigenvalues to find')
    parser.add_argument('-i', '--niter', dest='n_iter', type=int, help='Number of power iterations for random algorithm')
    parser.add_argument('-s', '--slepsc', dest='slepsc_flag', action='store_true', help='Use slepsc instead of random algorithm')
    parser.add_argument('-m', '--msft_flag', dest='msft_flag', action='store_true', help='Consider only the misfit term of the cost function without regularization')
    parser.add_argument('-p', '--parameters', dest='pflag', choices=[0, 1, 2], type=int, required=True, help='Inversion parameters: alpha (0), beta (1), alpha and beta (2)')

    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')

    parser.set_defaults(n_iter=1, slepsc_flag=False, msft_flag=False, outdir=False)
    args = parser.parse_args()

    num_eig = args.num_eig
    n_iter = args.n_iter
    slepsc_flag = args.slepsc_flag
    msft_flag = args.msft_flag
    pflag = args.pflag
    outdir = args.outdir
    dd = args.dd

    if not outdir:
        outdir = ''.join(['./run_eigendec_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    main(num_eig, n_iter, slepsc_flag, msft_flag, pflag, outdir, dd)
