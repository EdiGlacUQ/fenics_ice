import sys
sys.path.insert(0,'../code/')
import os
import argparse
from fenics import *
import model
import solver
import numpy as np
import time
import datetime
import pickle
from IPython import embed


def main(outdir, dd, periodic_bc, nx, ny, sl):

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
    Qp = Q

    # Make necessary modification for periodic bc
    if periodic_bc:
        mesh_length = np.NaN

        # If we're on a new mesh
        if nx and ny:
            L1 = xlim[-1] - xlim[0]
            L2 = ylim[-1] - ylim[0]
            assert( L1==L2), 'Periodic Boundary Conditions require a square domain'
            mesh_length = L1

        # If previous run   
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
    


    data_mask = Function(M,os.path.join(dd,data_mask_file))

    bed = Function(Q,os.path.join(dd,'bed.xml'))
    bmelt = Function(M,os.path.join(dd,'bmelt.xml'))
    smb = Function(M,os.path.join(dd,'smb.xml'))
    thick = Function(M,os.path.join(dd,'thick.xml'))
    alpha = Function(Qp,os.path.join(dd,'alpha.xml'))

#############################
    # #Load Data
    # data_mesh = Mesh(os.path.join(dd,'mesh.xml'))
    # mesh = data_mesh

    # M = FunctionSpace(data_mesh, 'DG', 0)
    # Mp = M
    # Q = FunctionSpace(data_mesh, 'Lagrange', 1) if os.path.isfile(os.path.join(dd,'param.p')) else M
    # Qp = Q

    # bed = Function(Q,os.path.join(dd,'bed.xml'))
    # bmelt = Function(M,os.path.join(dd,'bmelt.xml'))
    # smb = Function(M,os.path.join(dd,'smb.xml'))
    # thick = Function(M,os.path.join(dd,'thick.xml'))
    # mask = Function(M,os.path.join(dd,'mask.xml'))
    # alpha = Function(Qp,os.path.join(dd,'alpha.xml'))


    # if not os.path.isfile(os.path.join(dd,'param.p')):
    #     print('Generating new mesh')
    #     #Generate model mesh
    #     gf = 'grid_data.npz'
    #     npzfile = np.load(os.path.join(dd,'grid_data.npz'))
    #     xlim = npzfile['xlim']
    #     ylim = npzfile['ylim']

    #     if not nx:
    #         nx = int(npzfile['nx'])
    #     if not ny:
    #         ny = int(npzfile['ny'])

    #     mesh = RectangleMesh(Point(xlim[0],ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny)
    # else:
    #     print('Identified as previous run, reusing mesh')
    #     assert(not nx), 'Cannot change mesh resolution from previous run'
    #     assert(not ny), 'Cannot change mesh resolution from previous run'

    # if periodic_bc:
    #     if os.path.isfile(os.path.join(dd,'param.p')):
    #         periodic_bc = pickle.load(open(os.path.join(dd,'param.p'), 'rb'))['periodic_bc']
    #         assert(periodic_bc), 'Need to run periodic bc using original files'
    #     else:
    #         L1 = xlim[-1] - xlim[0]
    #         L2 = ylim[-1] - ylim[0]
    #         assert( L1==L2), 'Periodic Boundary Conditions require a square domain'
    #         periodic_bc = L1

    #     Qp = FunctionSpace(mesh,'Lagrange',1,constrained_domain=model.PeriodicBoundary(periodic_bc))
############################


    #Initialize Model
    param = {
            'outdir' : outdir,
            'periodic_bc': mesh_length,
            'sliding_law': sl
            }


    mdl = model.model(mesh,data_mask, param)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(data_mask)
    mdl.init_bmelt(bmelt)
    mdl.init_smb(smb)
    mdl.init_alpha(alpha)
    mdl.label_domain()


    if os.path.isfile(os.path.join(dd,'Bglen.xml')):
        Bglen = Function(M,os.path.join(dd,'Bglen.xml'))
        mdl.init_beta(mdl.bglen_to_beta(Bglen))

    else:
        print('Using default bglen (constant)')



    #Forward Solve
    slvr = solver.ssa_solver(mdl)
    slvr.def_mom_eq()
    slvr.solve_mom_eq()




    #Output model variables in ParaView+Fenics friendly format
    outdir = mdl.param['outdir']
    pickle.dump( mdl.param, open( os.path.join(outdir,'param.p'), "wb" ) )

    File(os.path.join(outdir,'mesh.xml')) << mdl.mesh

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

    vtkfile = File(os.path.join(outdir,'alpha.pvd'))
    xmlfile = File(os.path.join(outdir,'alpha.xml'))
    vtkfile << slvr.alpha
    xmlfile << slvr.alpha

    vtkfile = File(os.path.join(outdir,'Bglen.pvd'))
    xmlfile = File(os.path.join(outdir,'Bglen.xml'))
    Bglen = project(slvr.beta_to_bglen(slvr.beta),mdl.M)
    vtkfile << Bglen
    xmlfile << Bglen

    vtkfile = File(os.path.join(outdir,'surf.pvd'))
    xmlfile = File(os.path.join(outdir,'surf.xml'))
    vtkfile << mdl.surf
    xmlfile << mdl.surf

    vtkfile = File(os.path.join(outdir,'bmelt.pvd'))
    xmlfile = File(os.path.join(outdir,'bmelt.xml'))
    vtkfile << mdl.bmelt
    xmlfile << mdl.bmelt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--boundaries', dest='periodic_bc', action='store_true', help='Periodic boundary conditions')
    parser.add_argument('-x', '--cells_x', dest='nx', type=int, help='Number of cells in x direction')
    parser.add_argument('-y', '--cells_y', dest='ny', type=int, help='Number of cells in y direction')
    parser.add_argument('-q', '--slidinglaw', dest='sl', type=int,  help = 'Sliding Law (0: linear (default), 1: weertman)')
    
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')
   

    parser.set_defaults(periodic_bc=False,nx=False,ny=False, sl=0)
    args = parser.parse_args()

    outdir = args.outdir
    dd = args.dd
    periodic_bc = args.periodic_bc
    nx = args.nx
    ny = args.ny
    sl = args.sl

    if not outdir:
        outdir = ''.join(['./run_momsolve_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)



    main(outdir, dd, periodic_bc, nx, ny, sl)
