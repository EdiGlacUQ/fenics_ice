import sys
sys.path.insert(0,'../code/')
sys.path.insert(0,'../../dolfin_adjoint_custom/python/')

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


def main(outdir, dd, eigendir, lamfile, vecfile):

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


    delta = param['rc_inv'][1]
    gamma = param['rc_inv'][3]

    reg_op = prior.laplacian(delta, gamma, alpha.function_space())


    space = alpha.function_space()
    x, y = Function(alpha.function_space()), Function(alpha.function_space())
    z = Function(alpha.function_space())


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

    lam = np.maximum(lam,0)
    D = np.diag(lam / (lam + 1))



    W = np.zeros((x.vector().size(),nlam))
    with HDF5File(mpi_comm_world(), os.path.join(eigendir, vecfile), 'r') as hdf5data:
        for i in range(nlam):
            hdf5data.read(x, f'v/vector_{i}')
            v = x.vector().array()
            reg_op.action(x.vector(), y.vector())
            #mass.mult(y.vector(), z.vector())
            #tmp = z.vector.array()
            tmp = y.vector().array()
            sc = np.sqrt(np.dot(v,tmp))
            W[:,i] = v/sc


    hdf5data = HDF5File(mpi_comm_world(), os.path.join(dd, 'dJ_ts.h5'), 'r')


    dJ_cntrl = Function(space)

    run_length = param['run_length']
    num_sens = param['num_sens']
    t_sens = run_length if num_sens == 1 else np.linspace(0, run_length,num_sens)
    sigma = np.zeros(num_sens)


    for j in range(num_sens):
        hdf5data.read(dJ_cntrl, f'dJ/vector_{j}')

        tmp1 = np.dot(W.T,dJ_cntrl.vector().array())
        tmp2 = np.dot(D,tmp1 )
        P1 = np.dot(W,tmp2)

        reg_op.inv_action(dJ_cntrl.vector(),x.vector())
        P2 = x.vector().array()

        P = P2-P1
        variance = np.dot(dJ_cntrl.vector().array(), P2)
        sigma[j] = np.sqrt(variance)


    #Test that eigenvectors are prior inverse orthogonal
    # y.vector().set_local(W[:,398])
    # y.vector().apply('insert')
    # reg_op.action(y.vector(), x.vector())
    # #mass.mult(x.vector(),z.vector())
    # q = np.dot(y.vector().array(),x.vector().array())


    #Output model variables in ParaView+Fenics friendly format
    pickle.dump( [sigma, t_sens], open( os.path.join(outdir,'sigma_prior.p'), "wb" ) )



if __name__ == "__main__":
    stop_annotating()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', dest='pflag', choices=[0, 1, 2], type=int, required=True, help='Inversion parameters: alpha (0), beta (1), alpha and beta (2)')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')
    parser.add_argument('-l', '--lamfile', dest='lamfile', type=str, required=True, help = 'Pickle storing eigenvals')
    parser.add_argument('-k', '--vecfile', dest='vecfile', type=str, help = 'Hd5 File storing eigenvecs')
    parser.add_argument('-e', '--eigdir', dest='eigendir', type=str, required=True, help = 'Directory storing eigenpars')

    parser.set_defaults(outdir=False, vecfile = 'vr.h5')
    args = parser.parse_args()

    pflag = args.pflag
    outdir = args.outdir
    dd = args.dd
    eigendir = args.eigendir
    vecfile = args.vecfile
    lamfile = args.lamfile

    if not outdir:
        outdir = ''.join(['./run_tmp_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)



    main(outdir, dd, eigendir, lamfile, vecfile)
