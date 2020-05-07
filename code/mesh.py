"""
Module to handle all things mesh to avoid code repetition in run scripts.
"""

import os
from fenics import *
from dolfin import *
import numpy as np
import model

def get_mesh(params):
    """
    Return filenames for mesh, data_mesh & data_mask
    either 1) read from file or 2) created on-the-fly, depending
    on configuration from params.
    """

    nx = params.mesh.nx
    ny = params.mesh.ny
    dd = params.io.input_dir

    if nx and ny:

        #Generate model mesh
        print('Generating new mesh')
        gf = 'grid_data.npz' #TODO - unhardcode this
        npzfile = np.load(os.path.join(dd, gf))
        xlim = npzfile['xlim']
        ylim = npzfile['ylim']

        mesh_out = RectangleMesh(Point(xlim[0], ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny)

    # Reuse a mesh; in this case, mesh and data_mesh will be identical
    else:

        meshfile = os.path.join(dd, params.mesh.mesh_filename)

        assert os.path.isfile(meshfile), "Mesh file '%s' not found" % meshfile
        mesh_out = Mesh(meshfile)

    return mesh_out

def get_data_mesh(params):
    """
    Fetch the data mesh from file
    """
    dd = params.io.input_dir
    data_mesh_filename = params.mesh.data_mesh_filename
    data_mesh_file = os.path.join(dd, data_mesh_filename)
    assert os.path.isfile(data_mesh_file), "Data mesh file '%s' not found" % data_mesh_file
    return Mesh(data_mesh_file)

def get_data_mask(params, space):
    """
    Fetch the data mask from file
    """
    dd = params.io.input_dir
    data_mask_filename = params.mesh.data_mask_filename
    data_mask_file = os.path.join(dd, data_mask_filename)
    assert os.path.isfile(data_mask_file), "Data mask file '%s' not found" % data_mask_file

    return Function(space, data_mask_file)


def get_mesh_length(mesh):
    """
    Return a scalar mesh length (i.e. square mesh - isimp only!)
    """

    xmin, ymin = np.min(mesh.coordinates(), axis=0)
    xmax, ymax = np.max(mesh.coordinates(), axis=0)

    L1 = xmax - xmin
    L2 = ymax - ymin
    assert L1 == L2, 'Periodic Boundary Conditions require a square domain'

    mesh_length = L1
    return mesh_length

def setup_periodic_bc(params, mesh):
    """
    Return a FunctionSpace w/ periodic boundary
    """

    mesh_length = get_mesh_length(mesh)

    #TODO - I've replaced some rather complicated logic here (see below), is this OK?
    #TODO - what about parallel?

    # # If we're on a new mesh
    # if nx and ny:
    #     L1 = xlim[-1] - xlim[0]
    #     L2 = ylim[-1] - ylim[0]
    #     assert( L1==L2), 'Periodic Boundary Conditions require a square domain'
    #     mesh_length = L1

    # # If previous run   
    # elif os.path.isfile(os.path.join(dd,'param.p')):
    #     mesh_length = pickle.load(open(os.path.join(dd,'param.p'), 'rb'))['periodic_bc']
    #     assert(mesh_length), 'Need to run periodic bc using original files'

    # # Assume we're on a data_mesh
    # else:
    #     gf = 'grid_data.npz'
    #     npzfile = np.load(os.path.join(dd,'grid_data.npz'))
    #     xlim = npzfile['xlim']
    #     ylim = npzfile['ylim']
    #     L1 = xlim[-1] - xlim[0]
    #     L2 = ylim[-1] - ylim[0]
    #     assert( L1==L2), 'Periodic Boundary Conditions require a square domain'
    #     mesh_length = L1

    periodic_space = FunctionSpace(
        mesh,
        'Lagrange',
        1,
        constrained_domain=model.PeriodicBoundary(mesh_length)
    )

    return periodic_space
