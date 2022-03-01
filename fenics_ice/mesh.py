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

from .backend import FunctionSpace, MPI, Mesh, MeshFunction, \
    MeshValueCollection, VectorFunctionSpace, XDMFFile, parameters

import os
import numpy as np
from fenics_ice import model
from pathlib import Path
import logging

def get_mesh(params):
    """
    Gets mesh from file
    """

    dd = params.io.input_dir
    mesh_filename = params.mesh.mesh_filename
    meshfile = Path(dd) / mesh_filename
    filetype = meshfile.suffix

    #Ghost elements for DG in parallel
    parameters['ghost_mode'] = 'shared_facet'

    assert mesh_filename
    assert meshfile.exists(), "Mesh file '%s' not found" % meshfile

    if filetype == '.xml':
        mesh_in = Mesh(str(meshfile))

    elif filetype == '.xdmf':
        mesh_in = Mesh()
        mesh_xdmf = XDMFFile(MPI.comm_world, str(meshfile))
        mesh_xdmf.read(mesh_in)

    else:
        raise ValueError("Don't understand the mesh filetype: %s" % meshfile.name)

    return mesh_in

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
    if(dim == 1):
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

def get_ff_from_file(params, model, fill_val=0):
    """
    Return a FacetFunction defining the boundary conditions of the mesh.

    Expects to find an XDMF file containing a MeshValueCollection (sparse).
    Builds a 1D MeshFunction (i.e. FacetFunction) from this, filling missing
    values with fill_val.
    """

    dim = model.mesh.geometric_dimension()

    dd = params.io.input_dir
    ff_filename = Path(params.mesh.bc_filename)
    ff_file = dd/ff_filename

    assert ff_file.suffix == ".xdmf"
    assert ff_file.exists(), f"MeshValueCollection file {ff_file} not found"

    # Read the MeshValueCollection (sparse)
    ff_mvc = MeshValueCollection("size_t", model.mesh, dim=dim-1)
    ff_xdmf = XDMFFile(MPI.comm_world, str(ff_file))
    ff_xdmf.read(ff_mvc)

    # Create FacetFunction filled w/ default
    ff = MeshFunction('size_t', model.mesh, dim-1, int(fill_val))
    ff_arr = ff.array()

    # Get cell/facet topology
    model.mesh.init(dim, dim-1)
    connectivity = model.mesh.topology()(dim, dim-1)

    # Set ff from sparse mvc
    mvc_vals = ff_mvc.values()
    for ci_lei, value in mvc_vals.items():
        cell_index, local_entity_index = ci_lei
        entity_index = connectivity(cell_index)[local_entity_index]
        ff_arr[entity_index] = value

    return ff
