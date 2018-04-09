#!/usr/bin/env python

import sys
import os
import getopt
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

def main(argv):

    #Default Settings
    num_eig = False         #Number of eigenvalues to solve for. Non-optional argument.
    n_iter = 1              #Number of power iterations (randomized method only)
    outdir = '.'            #Directory to save output
    slepsc_flag=False       #Flag to use slepsc instead of randomized eigenvalue method
    msft_flag=False         #Consider only the misfit term in the cost function

    #Handle command line options to update default settings
    try:
      opts, args = getopt.getopt(argv,'smn:i:o:')
    except getopt.GetoptError:
      print 'file.py -n <number of eigenvalues> -i <power iterations>'
      sys.exit(2)
    for opt, arg in opts:
        if opt == '-n':
            num_eig = int(arg)
        elif opt == '-i':
            n_iter = int(arg)
        elif opt == '-s':
            slepsc_flag = True
        elif opt == '-m':
            msft_flag = True
        elif opt == '-o':
            outdir = arg
            if not os.path.isdir(outdir):
                print("Directory not valid, or does not exist")
                sys.exit(2)

    #Ensure user has provided the number of eigenvalues
    if not num_eig:
        print 'Use -n <number of eigenvalues>'
        sys.exit(2)

    #Data file: Should be previously completed inversion
    dd = './output_smith_inv/'

    #Load parameters of run
    param = pickle.load( open( ''.join([dd,'param.p']), "rb" ) )

    if msft_flag:
        embed()
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
        lam, v = eigendecomposition.eig(ddJw.ddJ_F.vector().local_size(), ddJw.apply, hermitian = True, N_eigenvalues = n_iter)
        pickle.dump( [lam,v], open( "{0}/slpesceig{1}.p".format(outdir,num_eig), "wb" ))

    else:
    #Determine eigenvalues using a randomized method
        A= eigenfunc.HessWrapper(slvr.ddJ,slvr.alpha)
        [lam,v] = eigenfunc.eigens(A,k=num_eig,n_iter=n_iter)
        pickle.dump( [lam,v], open( "{0}/randeig{1}.p".format(outdir,num_eig), "wb" ))

if __name__ == "__main__":
   main(sys.argv[1:])
