from fenics import *

import numpy as np
import sys
import os
from pylab import plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fenics import *
from IPython import embed
import model
import argparse


def main(dd,noise_sdev, bflag, L, nx, ny):

    mesh = Mesh(os.path.join(dd,'mesh.xml'))
    data_mesh = mesh

    if nx or ny:
        #Generate new data mesh
        print('Generating new mesh')
        npzfile = np.load(os.path.join(dd,'grid_data.npz'))
        xlim = npzfile['xlim']
        ylim = npzfile['ylim']

        data_mesh = RectangleMesh(Point(xlim[0],ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny)

    if bflag:
        V = VectorFunctionSpace(mesh, 'Lagrange', 1, dim=2, constrained_domain=model.PeriodicBoundary(L))
    else:
        V = VectorFunctionSpace(mesh, 'Lagrange', 1, dim=2)


    U = Function(V,os.path.join(dd,'U.xml'))
    M = FunctionSpace(data_mesh, 'DG', 0)

    N = Function(M)

    uu,vv = U.split(True)
    u = project(uu,M)
    v = project(vv,M)

    u_array = u.vector().get_local()
    v_array = v.vector().get_local()

    u_noise = np.random.normal(scale=noise_sdev, size=u_array.size)
    v_noise = np.random.normal(scale=noise_sdev, size=v_array.size)

    u.vector().set_local(u.vector().get_local() + u_noise)
    v.vector().set_local(v.vector().get_local() + v_noise)


    File(os.path.join(dd,'data_mesh.xml')) << data_mesh

    xmlfile = File(os.path.join(dd,'u_obs.xml'))
    xmlfile << u
    xmlfile = File(os.path.join(dd,'v_obs.xml'))
    xmlfile << v

    N.assign(Constant(noise_sdev))
    xmlfile = File(os.path.join(dd,'u_std.xml'))
    xmlfile << N

    xmlfile = File(os.path.join(dd,'v_std.xml'))
    xmlfile << N

    N.assign(Constant(1.0))
    xmlfile = File(os.path.join(dd,'mask_vel.xml'))
    xmlfile << N




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Directory with input data')
    parser.add_argument('-s', '--sigma', dest='noise_sdev', type=float,  help = 'Standard deviation of added Gaussian Noise')
    parser.add_argument('-b', '--boundaries', dest='bflag', action='store_true', help='Periodic boundary conditions')
    parser.add_argument('-L', '--length', dest='L', type=int, help='Length of IsmipC domain.')
    parser.add_argument('-x', '--cells_x', dest='nx', type=int, help='Number of cells in x direction')
    parser.add_argument('-y', '--cells_y', dest='ny', type=int, help='Number of cells in y direction')

    parser.set_defaults(noise_sdev = 1.0, bflag = False, L = False, nx = False, ny = False)
    args = parser.parse_args()

    dd = args.dd
    noise_sdev = args.noise_sdev
    bflag = args.bflag
    L = args.L
    nx = args.nx
    ny = args.ny

    if bflag and not L:
        print('Periodic boundary conditions requiring specifying the domain length with -L')
        raise SystemExit

    if (nx or ny):
        print('Regridding only works on square domains presently (i.e. IsmipC)')


    main(dd, noise_sdev, bflag, L, nx, ny)


