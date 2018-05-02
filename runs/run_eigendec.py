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
    param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )

    if msft_flag:
        rc_inv2 = param['rc_inv']
        rc_inv2[1:] = [0 for i in rc_inv2[1:]]
        param['rc_inv'] = rc_inv2

    #Complete Mesh and data mask
    data_mesh = Mesh(os.path.join(dd,'data_mesh.xml'))
    M_dm = FunctionSpace(data_mesh,'DG',0)
    data_mask = Function(M_dm,os.path.join(dd,'data_mask.xml'))

    #Ice only mesh
    mdl_mesh = Mesh(os.path.join(dd,'mesh.xml'))

    #Set up Function spaces
    V = VectorFunctionSpace(mdl_mesh,'Lagrange',1,dim=2)
    Q = FunctionSpace(mdl_mesh,'Lagrange',1)
    M = FunctionSpace(mdl_mesh,'DG',0)

    #Load fields
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
    Bglen = Function(M,os.path.join(dd,'Bglen.xml'))
    B2 = Function(Q,os.path.join(dd,'B2.xml'))

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
    A = eigendecomposition.HessWrapper(slvr.ddJ,slvr.alpha)

    if slepsc_flag:
        lam, v = eigendecomposition.slepsceig(A.xfn.vector().local_size(), A.apply, hermitian = True, N_eigenvalues = num_eig)
        fo = 'slepceig{0}{1}_{2}.p'.format(num_eig, 'm' if msft_flag else '', timestamp)
    else:
        lam,v = eigendecomposition.randeig(A,k=num_eig,n_iter=n_iter)
        fo = 'randeig{0}{1}_{2}.p'.format(num_eig, 'm' if msft_flag else '', timestamp)

    pickle.dump( [lam,v,num_eig, n_iter, slepsc_flag, msft_flag, outdir, dd], open( os.path.join(outdir, fo), "wb" ))


    #Sanity checks on eigenvalues/eigenvectors.
    neg_ind = np.nonzero(lam<0)[0]
    print('Number of eigenvalues: {0}'.format(num_eig))
    print('Number of negative eigenvalues: {0}'.format(len(neg_ind)))

    cntr_nn = 0
    cntr_np = 0

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
    print('The sign of {0} eigenvals is not supported by independent calculation using its eigenvector'.format(cntr_np))


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

    parser.set_defaults(n_iter=1, slepsc_flag=False, msft_flag=False, outdir='.')
    args = parser.parse_args()

    num_eig = args.num_eig
    n_iter = args.n_iter
    slepsc_flag = args.slepsc_flag
    msft_flag = args.msft_flag
    outdir = args.outdir
    dd = args.dd

    if not outdir:
        outdir = ''.join(['./run_eigendec_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    main(num_eig, n_iter, slepsc_flag, msft_flag, outdir, dd)
