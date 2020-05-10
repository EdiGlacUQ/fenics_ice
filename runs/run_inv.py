import sys
import os
import argparse
from dolfin import *
from tlm_adjoint_fenics import *

import model
import solver
import matplotlib as mpl
mpl.use("Agg")
import mesh as fice_mesh
from config import *
import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
import time
import datetime
import pickle
from IPython import embed


def run_inv(config_file):
    """
    Run the inversion part of the simulation
    """

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

    bed = Function(Q,os.path.join(dd,'bed.xml'))

    thick = Function(M,os.path.join(dd,'thick.xml'))
    u_obs = Function(M,os.path.join(dd,'u_obs.xml'))
    v_obs = Function(M,os.path.join(dd,'v_obs.xml'))
    u_std = Function(M,os.path.join(dd,'u_std.xml'))
    v_std = Function(M,os.path.join(dd,'v_std.xml'))
    mask_vel = Function(M,os.path.join(dd,'mask_vel.xml'))
    Bglen = Function(M,os.path.join(dd,'Bglen.xml'))
    bmelt = Function(M,os.path.join(dd,'bmelt.xml'))
    smb = Function(M,os.path.join(dd,'smb.xml'))

    pts_lengthscale = params.obs.pts_len

    mdl = model.model(mesh,data_mask, params)
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(data_mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std, pts_lengthscale)
    mdl.init_lat_dirichletbc()
    mdl.init_bmelt(bmelt)
    mdl.init_smb(smb)
    mdl.label_domain()

    mdl.gen_alpha()
    mdl.init_beta(mdl.bglen_to_beta(Bglen))            #Comment to use uniform Bglen

    #Next line will output the initial guess for alpha fed into the inversion
    #File(os.path.join(outdir,'alpha_initguess.pvd')) << mdl.alpha

    #Inversion
    slvr = solver.ssa_solver(mdl)
    slvr.inversion()


    #Output model variables in ParaView+Fenics friendly format
    outdir = params.io.output_dir

    #TODO - is this used to pass info between run_ parts?
    #pickle.dump( mdl.param, open( os.path.join(outdir,'param.p'), "wb" ) )

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


    vtkfile = File(os.path.join(outdir,'mask_vel.pvd'))
    xmlfile = File(os.path.join(outdir,'mask_vel.xml'))
    vtkfile << mdl.mask_vel
    xmlfile << mdl.mask_vel

    vtkfile = File(os.path.join(outdir,'u_obs.pvd'))
    xmlfile = File(os.path.join(outdir,'u_obs.xml'))
    vtkfile << mdl.u_obs
    xmlfile << mdl.u_obs

    vtkfile = File(os.path.join(outdir,'v_obs.pvd'))
    xmlfile = File(os.path.join(outdir,'v_obs.xml'))
    vtkfile << mdl.v_obs
    xmlfile << mdl.v_obs

    vtkfile = File(os.path.join(outdir,'u_std.pvd'))
    xmlfile = File(os.path.join(outdir,'u_std.xml'))
    vtkfile << mdl.u_std
    xmlfile << mdl.u_std

    vtkfile = File(os.path.join(outdir,'v_std.pvd'))
    xmlfile = File(os.path.join(outdir,'v_std.xml'))
    vtkfile << mdl.v_std
    xmlfile << mdl.v_std

    vtkfile = File(os.path.join(outdir,'uv_obs.pvd'))
    xmlfile = File(os.path.join(outdir,'uv_obs.xml'))
    U_obs = project((mdl.v_obs**2 + mdl.u_obs**2)**(1.0/2.0), mdl.M)
    vtkfile << U_obs
    xmlfile << U_obs

    vtkfile = File(os.path.join(outdir,'alpha.pvd'))
    xmlfile = File(os.path.join(outdir,'alpha.xml'))
    vtkfile << slvr.alpha
    xmlfile << slvr.alpha

    vtkfile = File(os.path.join(outdir,'Bglen.pvd'))
    xmlfile = File(os.path.join(outdir,'Bglen.xml'))
    Bglen = project(slvr.beta_to_bglen(slvr.beta),mdl.M)
    vtkfile << Bglen
    xmlfile << Bglen

    vtkfile = File(os.path.join(outdir,'bmelt.pvd'))
    xmlfile = File(os.path.join(outdir,'bmelt.xml'))
    vtkfile << slvr.bmelt
    xmlfile << slvr.bmelt

    vtkfile = File(os.path.join(outdir,'smb.pvd'))
    xmlfile = File(os.path.join(outdir,'smb.xml'))
    vtkfile << slvr.smb
    xmlfile << slvr.smb

    vtkfile = File(os.path.join(outdir,'surf.pvd'))
    xmlfile = File(os.path.join(outdir,'surf.xml'))
    vtkfile << mdl.surf
    xmlfile << mdl.surf

    pickle.dump( mdl.uv_obs_pts, open( os.path.join(outdir,'obs_pts.p'), "wb" ) )

if __name__ == "__main__":
    stop_annotating()
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_inv(sys.argv[1])
