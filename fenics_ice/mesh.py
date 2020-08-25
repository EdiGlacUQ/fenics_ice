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

"""
Module to handle all things mesh to avoid code repetition in run scripts.
"""

import os
from fenics import *
from dolfin import *
import numpy as np
from fenics_ice import model
import logging

def get_mesh(params):
    """
    Gets mesh from file
    """

    dd = params.io.input_dir
    mesh_filename = params.mesh.mesh_filename
    meshfile = os.path.join(dd, mesh_filename)

    #Ghost elements for DG in parallel
    parameters['ghost_mode'] = 'shared_facet'

    assert mesh_filename
    assert os.path.isfile(meshfile), "Mesh file '%s' not found" % meshfile

    mesh_out = Mesh(meshfile)

    return mesh_out

def get_mesh_length(mesh):
    """
    Return a scalar mesh length (i.e. square mesh - isimp only!)
    """

    xmin, ymin = np.min(mesh.coordinates(), axis=0)
    xmax, ymax = np.max(mesh.coordinates(), axis=0)

    comm = mesh.mpi_comm()

    xmin = MPI.min(comm, xmin)
    xmax = MPI.max(comm, xmax)
    ymin = MPI.min(comm, ymin)
    ymax = MPI.max(comm, ymax)

    L1 = xmax - xmin
    L2 = ymax - ymin
    assert L1 == L2, 'Periodic Boundary Conditions require a square domain'

    return L1

def get_periodic_space(params, mesh, deg=1, dim=1):
    """
    Return a Lagrange FunctionSpace w/ periodic boundary
    """

    mesh_length = get_mesh_length(mesh)

    # Periodic BCs don't work with ghost_mode = shared_facet...
    # https://fenicsproject.discourse.group/t/use-mpirun-for-dg-with-periodic-bc/2846
    # Seems like a common issue
    if(dim==1):
        periodic_space = FunctionSpace(
            mesh,
            'Lagrange',
            deg,
            dim,
            constrained_domain=model.PeriodicBoundary(mesh_length)
        )
    else:
        periodic_space = VectorFunctionSpace(
            mesh,
            'Lagrange',
            deg,
            dim,
            constrained_domain=model.PeriodicBoundary(mesh_length)
        )

    return periodic_space
