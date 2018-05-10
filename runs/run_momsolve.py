import sys
sys.path.insert(0,'../code/')
import os
import argparse
from fenics import *
import model
import solver
import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
import time
import datetime
import pickle
from IPython import embed


def main(outdir, dd, bflag, nx, ny):

    #Load Data
    data_mesh = Mesh(os.path.join(dd,'mesh.xml'))
    mesh = data_mesh

    M = FunctionSpace(data_mesh, 'DG', 0)
    Q = FunctionSpace(data_mesh, 'Lagrange', 1) if os.path.isfile(os.path.join(dd,'param.p')) else M

    bed = Function(Q,os.path.join(dd,'bed.xml'))
    B2 = Function(M,os.path.join(dd,'B2.xml'))
    Bglen = Function(M,os.path.join(dd,'Bglen.xml'))
    bmelt = Function(M,os.path.join(dd,'bmelt.xml'))
    thick = Function(M,os.path.join(dd,'thick.xml'))
    mask = Function(M,os.path.join(dd,'mask.xml'))


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
            'periodic_bc': bflag
            }

    mdl = model.model(mesh,mask, param)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(mask)
    mdl.init_bmelt(bmelt)
    mdl.init_alpha(mdl.apply_prmz(B2))
    mdl.init_beta(mdl.apply_prmz(Bglen), pert=False)
    mdl.label_domain()

    #Inversion
    slvr = solver.ssa_solver(mdl)
    slvr.def_mom_eq()
    slvr.solve_mom_eq()



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


    vtkfile = File(os.path.join(outdir,'alpha.pvd'))
    xmlfile = File(os.path.join(outdir,'alpha.xml'))
    vtkfile << mdl.alpha
    xmlfile << mdl.alpha

    vtkfile = File(os.path.join(outdir,'Bglen.pvd'))
    xmlfile = File(os.path.join(outdir,'Bglen.xml'))
    Bglen = project(mdl.rev_prmz(mdl.beta),mdl.M)
    vtkfile << Bglen
    xmlfile << Bglen

    vtkfile = File(os.path.join(outdir,'B2.pvd'))
    xmlfile = File(os.path.join(outdir,'B2.xml'))
    B2 = project(mdl.rev_prmz(mdl.alpha), mdl.M)
    vtkfile << B2
    xmlfile << B2

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
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')
    parser.add_argument('-b', '--boundaries', dest='bflag', action='store_true', help='Periodic boundary conditions')
    parser.add_argument('-x', '--cells_x', dest='nx', type=int, help='Number of cells in x direction (defaults to data resolution)')
    parser.add_argument('-y', '--cells_y', dest='ny', type=int, help='Number of cells in y direction (defaults to data resolution)')

    parser.set_defaults(bflag=False,nx=False,ny=False)
    args = parser.parse_args()

    outdir = args.outdir
    dd = args.dd
    bflag = args.bflag
    nx = args.nx
    ny = args.ny

    if not outdir:
        outdir = ''.join(['./run_momsolve_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)



    main(outdir, dd, bflag, nx, ny)
