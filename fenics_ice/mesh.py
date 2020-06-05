"""
Module to handle all things mesh to avoid code repetition in run scripts.
"""

import os
from fenics import *
from dolfin import *
import numpy as np
from fenics_ice import model
import logging

def create_ismip_mesh(params):
    """
    Create rectangular mesh as specified in params
    """

    nx = params.mesh.nx
    ny = params.mesh.ny
    dd = params.io.input_dir
    assert nx

    #Generate model mesh
    logging.info('Generating new mesh')
    gf = 'grid_data.npz' #TODO - unhardcode this
    npzfile = np.load(os.path.join(dd, gf))
    xlim = npzfile['xlim']
    ylim = npzfile['ylim']

    mesh_out = RectangleMesh(Point(xlim[0], ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny)

    return mesh_out

def get_mesh(params):
    """
    Gets mesh from file
    """

    dd = params.io.input_dir
    mesh_filename = params.mesh.mesh_filename
    meshfile = os.path.join(dd, mesh_filename)

    assert mesh_filename
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


def get_mask(params, space):
    """
    Fetch the mask associated with our run mesh
    """
    dd = params.io.output_dir
    mask_filename = "mask.xml" #params.mesh.data_mask_filename #TODO!!
    mask_file = os.path.join(dd, mask_filename)
    assert os.path.isfile(mask_file), "Data mask file '%s' not found" % mask_file

    return Function(space, mask_file)


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

def get_periodic_space(params, mesh, deg=1, dim=1):
    """
    Return a Lagrange FunctionSpace w/ periodic boundary
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