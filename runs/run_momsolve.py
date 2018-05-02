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


def main(outdir, dd, nx, ny):

    #Load Data
    data_mesh = Mesh(os.path.join(dd,'mesh.xml'))
    M = FunctionSpace(data_mesh, 'DG', 0)

    B2 = Function(M,os.path.join(dd,'B2.xml'))
    Bglen = Function(M,os.path.join(dd,'Bglen.xml'))
    bmelt = Function(M,os.path.join(dd,'bmelt.xml'))
    bed = Function(M,os.path.join(dd,'bed.xml'))
    thick = Function(M,os.path.join(dd,'thick.xml'))
    mask = Function(M,os.path.join(dd,'mask.xml'))

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

    #Initialize Model
    param = {
            'outdir' : outdir
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
    parser.add_argument('-x', '--cells_x', dest='nx', type=int, help='Number of cells in x direction (defaults to data resolution)')
    parser.add_argument('-y', '--cells_y', dest='ny', type=int, help='Number of cells in y direction (defaults to data resolution)')

    parser.set_defaults(nx=False,ny=False)
    args = parser.parse_args()

    outdir = args.outdir
    dd = args.dd
    nx = args.nx
    ny = args.ny

    if not outdir:
        outdir = ''.join(['./run_momsolve_', datetime.datetime.now().strftime("%m%d%H%M%S")])
        print('Creating output directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)



    main(outdir, dd, nx, ny)
