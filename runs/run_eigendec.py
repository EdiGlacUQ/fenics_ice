#!/usr/bin/env python

import sys
import os
import argparse
from fenics import *
from dolfin_adjoint import *
import pickle
from IPython import embed
sys.path.insert(0,'../code/')
import model
import solver
import optim
import eigenfunc
import eigendecomposition
import datetime
import numpy as np

def main(num_eig, n_iter, slepsc_flag, msft_flag, outdir, dd):

    #Load parameters of run
    param = pickle.load( open( ''.join([dd,'param.p']), "rb" ) )

    if msft_flag:
        rc_inv2 = param['rc_inv']
        rc_inv2[1:] = [0 for i in rc_inv2[1:]]
        param['rc_inv'] = rc_inv2

    #Complete Mesh and data mask
    data_mesh = Mesh(''.join([dd,'data_mesh.xml']))
    M_dm = FunctionSpace(data_mesh,'DG',0)
    data_mask = Function(M_dm,''.join([dd,'data_mask.xml']))

    #Ice only mesh
    mdl_mesh = Mesh(''.join([dd,'mesh.xml']))

    #Set up Function spaces
    V = VectorFunctionSpace(mdl_mesh,'Lagrange',1,dim=2)
    Q = FunctionSpace(mdl_mesh,'Lagrange',1)
    M = FunctionSpace(mdl_mesh,'DG',0)

    #Load fields
    U = Function(V,''.join([dd,'U.xml']))
    alpha = Function(Q,''.join([dd,'alpha.xml']))
    beta = Function(Q,''.join([dd,'beta.xml']))
    bed = Function(Q,''.join([dd,'bed.xml']))
    surf = Function(Q,''.join([dd,'surf.xml']))
    thick = Function(M,''.join([dd,'thick.xml']))
    mask = Function(M,''.join([dd,'mask.xml']))
    mask_vel = Function(M,''.join([dd,'mask_vel.xml']))
    u_obs = Function(M,''.join([dd,'u_obs.xml']))
    v_obs = Function(M,''.join([dd,'v_obs.xml']))
    u_std = Function(M,''.join([dd,'u_std.xml']))
    v_std = Function(M,''.join([dd,'v_std.xml']))
    uv_obs = Function(M,''.join([dd,'uv_obs.xml']))
    Bglen = Function(M,''.join([dd,'Bglen.xml']))
    B2 = Function(Q,''.join([dd,'B2.xml']))

    #Initialize our model object
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


    #Setup our solver object
    slvr = solver.ssa_solver(mdl)

    #Solve for velocities
    slvr.def_mom_eq()
    slvr.solve_mom_eq()

    #Set the inversion cost functional, and the Hessian w.r.t parameter
    slvr.set_J_inv()
    slvr.set_hessian_action(slvr.alpha)

    #Determine eigenvalues with slepsc using the interfacing script written by James Maddison
    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
    if slepsc_flag:
        class ddJ_wrapper(object):
            def __init__(self, ddJ_action, cntrl):
                self.ddJ_action = ddJ_action
                self.ddJ_F = Function(cntrl.function_space())

            def apply(self,x):
                self.ddJ_F.vector().set_local(x.getArray())
                self.ddJ_F.vector().apply('insert')
                return self.ddJ_action(self.ddJ_F).vector().get_local()


        ddJw = ddJ_wrapper(slvr.ddJ,slvr.alpha)
        lam, v = eigendecomposition.eig(ddJw.ddJ_F.vector().local_size(), ddJw.apply, hermitian = True, N_eigenvalues = num_eig)
        pickle.dump( [lam,v,num_eig, n_iter, slepsc_flag, msft_flag, outdir, dd], open( "{0}/slepceig{1}_{2}.p".format(outdir,num_eig,timestamp), "wb" ))
        A= eigenfunc.HessWrapper(slvr.ddJ,slvr.alpha) #for sanity checks
    else:
    #Determine eigenvalues using a randomized method
        A= eigenfunc.HessWrapper(slvr.ddJ,slvr.alpha)
        [lam,v] = eigenfunc.eigens(A,k=num_eig,n_iter=n_iter)
        pickle.dump( [lam,v,num_eig, n_iter, slepsc_flag, msft_flag, outdir, dd], open( "{0}/randeig{1}_{2}.p".format(outdir,num_eig,timestamp), "wb" ))


    #Sanity checks on eigenvalues/eigenvectors.
    neg_ind = np.nonzero(lam<0)[0]
    print('Number of eigenvalues: {0}'.format(num_eig))
    print('Number of negative eigenvalues: {0}'.format(len(neg_ind)))

    cntr_nn = 0.0
    cntr_np = 0.0

    print('Checking negative eigenvalues...')
    #For each negative eigenvalue, independently recalculate their value using its eigenvector
    for i in neg_ind:
        ev = v[:,i]
        ll = lam[i]
        ll2 = np.dot(ev, A.apply(ev)) / np.dot(ev,ev)

        #Compare signs of the calculated eigenvals, increment correct counter
        if np.sign(ll) == np.sign(ll2):
            cntr_nn += 1
        else:
            cntr_np +=1
            print('Eigenval {0} at index {1} is a false negative'.format(ll,i))

    print('The number of verified negative eigenvals is {0}'.format(cntr_nn))
    print('The sign {0} eigenvals is not supported by independent calculation using its eigenvector'.format(cntr_np))


    print('Checking the accuracy of eigenval/eigenvec pairs...')

    npair = min(num_eig, 10) #Select a maximum of 10 pairs
    f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)] #selector function
    pind = f(npair,num_eig)

    for i in pind:
        ev = v[:,i]
        ll = lam[i]
        res = np.linalg.norm(A.apply(ev) - ll*ev)/ll
        print('Residual of {0} at index {1}'.format(res, i))





if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--numeig', dest='num_eig', type=int, required=True, help='Number of eigenvalues to find')
    parser.add_argument('-i', '--niter', dest='n_iter', type=int, help='Number of power iterations for random algorithm')
    parser.add_argument('-s', '--slepsc', dest='slepsc_flag', action='store_true', help='Use slepsc instead of random algorithm')
    parser.add_argument('-m', '--msft_flag', dest='msft_flag', action='store_true', help='Consider only the misfit term of the cost function without regularization')

    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')

    parser.set_defaults(run_length=10.0, n_steps=120, init_yr=5, outdir='./')
    args = parser.parse_args()

    num_eig = args.num_eig
    n_iter = args.n_iter
    slepsc_flag = args.slepsc_flag
    msft_flag = args.msft_flag
    outdir = args.outdir
    dd = args.dd


    main(num_eig, n_iter, slepsc_flag, msft_flag, outdir, dd)
