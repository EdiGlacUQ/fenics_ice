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
import argparse
from pathlib import Path
from dolfin import *
from tlm_adjoint_fenics import *

from fenics_ice import model, solver, inout
from fenics_ice import mesh as fice_mesh
from fenics_ice.config import ConfigParser
import fenics_ice.fenics_util as fu

import matplotlib as mpl
#mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import pickle
from IPython import embed

def run_inv(config_file):
    """
    Run the inversion part of the simulation
    """

    # Read run config file
    params = ConfigParser(config_file)

    log = inout.setup_logging(params)
    inout.log_preamble("inverse", params)
    outdir = params.io.output_dir

    # Load the static model data (geometry, smb, etc)
    input_data = inout.InputData(params)

    # Get the model mesh
    mesh = fice_mesh.get_mesh(params)

    # TODO use this or get rid of it
    pts_lengthscale = params.obs.pts_len


    # Add random noise to Beta field iff we're inverting for it

    # Next line will output the initial guess for alpha fed into the inversion
    # File(os.path.join(outdir,'alpha_initguess.pvd')) << mdl.alpha

    #####################
    # Run the Inversion #
    #####################


    gamma_list = np.array([1.,2.,5.,8.,10.,12.,20.,50.,100.,200.,500.])

    Mis = np.zeros(len(gamma_list))
    Norm = np.zeros(len(gamma_list))

    for num,gamm in enumerate(gamma_list):
     print ('DOING INVERSION WITH GAMMA EQUALS ' + str(gamm))

     mdl = model.model(mesh, input_data, params)
     mdl.gen_alpha()
     mdl.bglen_from_data()
     mdl.init_beta(mdl.bglen_to_beta(mdl.bglen), params.inversion.beta_active)
     slvr = solver.ssa_solver(mdl)

     slvr.delta_alpha = 1.e-5
     slvr.gamma_alpha = gamm
     slvr.inversion()
    
     Jreg = slvr.comp_J_inv(verbose=False, noMisfit=True, noReg=False)
     Jmis = slvr.comp_J_inv(verbose=False, noMisfit=False, noReg=True)

     Mis[num] = Jmis.value()

     Norm[num] = Jreg.value()/gamm

#    fig = plt.figure()
#    ax = fig.add_subplot(111)

#    plt.plot(Norm,Mis,'+')

#    for num in range(len(gamma_list)):                                       # <--
#     ax.annotate('(%5.2f)' % gamma_list[num], xy=(Norm[num],Mis[num]), textcoords='data')
#    plt.show()
    np.save(outdir + '/LCurvNorm',Norm)
    np.save(outdir + '/LCurvMis',Mis)
    np.save(outdir + '/LCurvGam',gamma_list)
    print('done with L curve')
    

     


    ###########################
    #  Write out variables    #
    ###########################


if __name__ == "__main__":
    stop_annotating()
    assert len(sys.argv) == 2, "Expected a configuration file (*.toml)"
    run_inv(sys.argv[1])
