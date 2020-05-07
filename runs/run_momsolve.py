"""
Run the momentum solver only. Primarily used to generate velocity field
for ismip-c case before running the main model.
"""

import sys
sys.path.insert(0,'../code/')
import os
import argparse
from fenics import *
import model
import solver
import mesh as fice_mesh
from config import *
import numpy as np
import time
import datetime
import pickle
from IPython import embed


def main(config_file):

    #Read run config file
    params = ConfigParser(config_file)

    dd = params.io.input_dir

    # Determine Mesh
    mesh = fice_mesh.get_mesh(params)
    data_mesh = fice_mesh.get_data_mesh(params)

    # Define Function Spaces
    M = FunctionSpace(data_mesh, 'DG', 0)
    Q = FunctionSpace(data_mesh, 'Lagrange', 1)

    # Make necessary modification for periodic bc
    if params.mesh.periodic_bc:
        Qp = fice_mesh.setup_periodic_bc(params, data_mesh)
    else:
        Qp = Q


    data_mask = fice_mesh.get_data_mask(params, M)

    bed = Function(Q, os.path.join(dd, 'bed.xml'))
    bmelt = Function(M, os.path.join(dd, 'bmelt.xml'))
    smb = Function(M, os.path.join(dd, 'smb.xml'))
    thick = Function(M, os.path.join(dd, 'thick.xml'))
    alpha = Function(Qp, os.path.join(dd, 'alpha.xml'))

    mdl = model.model(mesh, data_mask, params)
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
    outdir = params.io.output_dir
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
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    main(sys.argv[1])
