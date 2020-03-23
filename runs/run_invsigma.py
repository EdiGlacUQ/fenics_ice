import sys
import os
import argparse
from dolfin import *
from tlm_adjoint import *

import model
import solver
import prior

import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
import time
import datetime
import pickle
from petsc4py import PETSc
from IPython import embed


def main(outdir, dd, eigendir, lamfile, vecfile, pflag, threshlam):

    param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )

    #Load Data
    data_mesh = Mesh(os.path.join(dd,'mesh.xml'))
    mesh = data_mesh

    #Set up Function spaces
    Q = FunctionSpace(mesh,'Lagrange',1)
    M = FunctionSpace(mesh,'DG',0)

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


    mdl = model.model(mesh,mask, param)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
    mdl.init_lat_dirichletbc()
    mdl.label_domain()
    mdl.init_alpha(alpha)

    rc_inv = param['rc_inv']
    if pflag == 0:
        delta = rc_inv[1]
        gamma = rc_inv[3]
    elif pflag == 1:
        delta = rc_inv[2]
        gamma = rc_inv[4]

    opts = {'0': mdl.alpha, '1': mdl.beta, '2': [mdl.alpha,mdl.beta]}
    cntrl = opts[str(pflag)]
    space = cntrl.function_space()

    sigma = Function(space)
    x, y = Function(space), Function(space)
    z = Function(space)

    reg_op = prior.laplacian(delta, gamma, space)


    test, trial = TestFunction(space), TrialFunction(space)
    mass = assemble(inner(test,trial)*dx)
    mass_solver = KrylovSolver("cg", "sor")
    mass_solver.parameters.update({"absolute_tolerance":1.0e-32,
                               "relative_tolerance":1.0e-14})
    mass_solver.set_operator(mass)


    with open(os.path.join(eigendir, lamfile), 'rb') as ff:
        eigendata = pickle.load(ff)
        lam = eigendata[0].real.astype(np.float64)
        nlam = len(lam)


    W = np.zeros((x.vector().size(),nlam))
    with HDF5File(MPI.comm_world, os.path.join(eigendir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            hdf5data.read(x, f'v/vector_{i}')
            v = x.vector().get_local()
            reg_op.action(x.vector(), y.vector())
            tmp = y.vector().get_local()
            sc = np.sqrt(np.dot(v,tmp))
            W[:,i] = v/sc


    pind = np.flatnonzero(lam>threshlam)
    lam = lam[pind]
    W = W[:,pind]

    D = np.diag(lam / (lam + 1))


    sigma_get_local = np.zeros(space.dim())
    ivec = np.zeros(space.dim())

    for j in range(sigma_get_local.size):

        ivec.fill(0)
        ivec[j] = 1.0
        y.vector().set_local(ivec)
        y.vector().apply('insert')

        tmp1 = np.dot(W.T,ivec)
        tmp2 = np.dot(D,tmp1 )
        P1 = np.dot(W,tmp2)

        reg_op.inv_action(y.vector(),x.vector())
        P2 = x.vector().get_local()

        P = P2-P1
        sigma_get_local[j] = np.sqrt(np.dot(ivec, P))

    sigma.vector().set_local(sigma_get_local)
    sigma.vector().apply('insert')

    vtkfile = File(os.path.join(outdir,'{0}_sigma.pvd'.format(cntrl.name()) ))
    xmlfile = File(os.path.join(outdir,'{0}_sigma.xml'.format(cntrl.name()) ))
    vtkfile << sigma
    xmlfile << sigma

if __name__ == "__main__":
    stop_annotating()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', dest='pflag', choices=[0, 1, 2], type=int, required=True, help='Inversion parameters: alpha (0), beta (1), alpha and beta (2)')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, required=True, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')
    parser.add_argument('-l', '--lamfile', dest='lamfile', type=str, required=True, help = 'Pickle file storing eigenvals')
    parser.add_argument('-k', '--vecfile', dest='vecfile', type=str, help = 'Hd5 File storing eigenvecs')
    parser.add_argument('-e', '--eigdir', dest='eigendir', type=str, required=True, help = 'Directory storing eigenpars')
    parser.add_argument('-c', '--threshlam', dest='threshlam', type=float, help = 'Threshold eigenvalue value for cutoff')

    parser.set_defaults(outdir=False, threshlam = 1e-1, vecfile = 'vr.h5')
    args = parser.parse_args()

    pflag = args.pflag
    outdir = args.outdir
    dd = args.dd
    eigendir = args.eigendir
    vecfile = args.vecfile
    lamfile = args.lamfile
    threshlam = args.threshlam

    if not outdir:
        outdir = ''.join(['./run_tmp_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)



    main(outdir, dd, eigendir, lamfile, vecfile, pflag, threshlam)
