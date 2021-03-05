# For fenics_ice copyright information see ACKNOWLEDGEMENTS in the fenics_ice
# root directory

# This file is part of fenics_ice.
#
# fenics_ice is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# fenics_ice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

import sys
import os
import getopt
import argparse

from fenics import *
from dolfin import *
import ufl

from fenics_ice import model, solver
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
import fenics_ice.fenics_util as fu

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import pickle
from IPython import embed

def main(config_file):

    print("===WARNING=== - this code has not been fully adapted to new config format")
    print("Consult TODOs in run_balancemeltrates.py")

    init_yr = 5 #TODO - where in the config?

    #Read run config file
    params = ConfigParser(config_file)

    log = inout.setup_logging(params)
    inout.log_preamble("balance meltrates", params)

    dd = params.io.input_dir
    outdir = params.io.output_dir
    run_length = params.time.run_length
    n_steps = params.time.total_steps

    assert init_yr < run_length

    # #Load Data
    # param = pickle.load( open( os.path.join(dd,'param.p'), "rb" ) )

    # param['outdir'] = outdir
    # param['sliding_law'] = sl
    # param['picard_params'] = {"nonlinear_solver":"newton",
    #             "newton_solver":{"linear_solver":"umfpack",
    #             "maximum_iterations":25,
    #             "absolute_tolerance":1.0e-3,
    #             "relative_tolerance":5.0e-2,
    #             "convergence_criterion":"incremental",
    #             "error_on_nonconvergence":False,
    #             "lu_solver":{"same_nonzero_pattern":False, "symmetric":False, "reuse_factorization":False}}}

    #Load Data
    mesh = Mesh(os.path.join(outdir, 'mesh.xml'))

    M = FunctionSpace(mesh, 'DG', 0)
    #TODO - what's the logic here?:
    Q = FunctionSpace(mesh, 'Lagrange', 1)# if os.path.isfile(os.path.join(dd,'param.p')) else M

    mask = Function(M,os.path.join(outdir,'mask.xml'))

    if os.path.isfile(os.path.join(outdir,'data_mesh.xml')):
        data_mesh = Mesh(os.path.join(outdir,'data_mesh.xml'))
        Mdata = FunctionSpace(data_mesh, 'DG', 0)
        data_mask = Function(Mdata, os.path.join(outdir,'data_mask.xml'))
    else:
        data_mesh = mesh
        data_mask = mask

    if not params.mesh.periodic_bc:
       Qp = Q
       V = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
    else:
       Qp = fice_mesh.get_periodic_space(params, mesh, dim=1)
       V = fice_mesh.get_periodic_space(params, mesh, dim=2)

    #Load fields
    U = Function(V,os.path.join(outdir,'U.xml'))

    alpha = Function(Qp,os.path.join(outdir,'alpha.xml'))
    beta = Function(Qp,os.path.join(outdir,'beta.xml'))
    bed = Function(Q,os.path.join(outdir,'bed.xml'))

    smb = Function(M,os.path.join(outdir, 'smb.xml'))
    thick = Function(M,os.path.join(outdir,'thick.xml'))
    mask_vel = Function(M,os.path.join(outdir,'mask_vel.xml'))
    u_obs = Function(M,os.path.join(outdir,'u_obs.xml'))
    v_obs = Function(M,os.path.join(outdir,'v_obs.xml'))
    u_std = Function(M,os.path.join(outdir,'u_std.xml'))
    v_std = Function(M,os.path.join(outdir,'v_std.xml'))
    uv_obs = Function(M,os.path.join(outdir,'uv_obs.xml'))

    mdl = model.model(mesh, data_mask, params, init_fields=False)  # TODO initialization
    mdl.init_bed(bed)
    mdl.init_thick(thick)
    mdl.gen_surf()
    mdl.init_mask(mask)
    mdl.init_vel_obs(u_obs,v_obs,mask_vel,u_std,v_std)
    mdl.init_lat_dirichletbc()
    mdl.init_bmelt(Constant(0.0))
    mdl.init_alpha(alpha)
    mdl.init_beta(beta, False)
    mdl.init_smb(smb)
    mdl.label_domain()

    #Solve
    slvr = solver.ssa_solver(mdl)
    slvr.save_ts_zero()
    slvr.timestep(save = 1, adjoint_flag=0)

    #Balance melt rates

    #Load time series of ice thicknesses
    hdf = HDF5File(slvr.mesh.mpi_comm(), os.path.join(outdir, 'H_ts.h5'), "r")
    attr = hdf.attributes("H")
    nsteps = attr['count']

    #model time step
    dt = params.time.dt

    #Model iterations to difference between
    iter_s = np.ceil(init_yr/dt)  #Iteration closest to 5yr
    iter_f = nsteps - 1         #Final iteration
    dT = dt*(iter_f - iter_s) #Time diff in years between iterations

    #Read iteration data
    HS = Function(slvr.M)
    HF = Function(slvr.M)
    hdf.read(HS, "H/vector_{0}".format(int(iter_s)))
    hdf.read(HF, "H/vector_{0}".format(int(iter_f)))

    #Mask out grounded region
    rhow = params.constants.rhow
    rhoi = params.constants.rhoi
    H_s = -rhow/rhoi * bed
    fl_ex = conditional(slvr.H_init <= H_s, Constant(1.0), Constant(0.0))

    #Calculate bmelt
    bmelt = project(ufl.Max(fl_ex*(HF - HS)/dT, Constant(0.0)), slvr.M)

    #Output model variables in ParaView+Fenics friendly format
    with open(os.path.join(outdir,'bmeltrate_param.p'), "wb" ) as bmeltfile:
        pickle.dump( mdl.param, bmeltfile)

    # File(os.path.join(outdir,'mesh.xml')) << mdl.mesh

    vtkfile = File(os.path.join(outdir,'bmelt.pvd'))
    xmlfile = File(os.path.join(outdir,'bmelt.xml'))
    vtkfile << bmelt
    xmlfile << bmelt

    return mdl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#    parser.add_argument('-r', '--runlength', dest='run_length', type=float, help='Length of forward run in years (Default 10yrs)')
#    parser.add_argument('-n', '--nsteps', dest='n_steps', type=int, help='Number of model timesteps (Default 240)')
#    parser.add_argument('-y', '--yearinitial', dest='init_yr', type=int, help='The initial year to difference final model results with to calculate balance melt rates (Default 5yrs)')
#    parser.add_argument('-q', '--slidinglaw', dest='sl', type=float,  help = 'Sliding Law (0: linear (default), 1: budd)')

#    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
#    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')

    # parser.set_defaults(run_length=10.0, n_steps=240, init_yr=5, outdir=False, sl=0)
    # args = parser.parse_args()

    # run_length = args.run_length
    # n_steps = args.n_steps
    # init_yr = args.init_yr
    # outdir = args.outdir
    # dd = args.dd
    # sl = args.sl


    # if init_yr >= run_length:
    #     print('Init year must less than the run length')
    #     sys.exit(2)

    # if not outdir:
    #     outdir = ''.join(['./balance_melt_rates_', datetime.datetime.now().strftime("%m%d%H%M%S")])
    #     print('Creating output directory: {0}'.format(outdir))
    #     os.makedirs(outdir)
    # else:
    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)


    # if init_yr >= run_length:
    #     print('Init year must less than the run length')
    #     sys.exit(2)

    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    main(sys.argv[1])
