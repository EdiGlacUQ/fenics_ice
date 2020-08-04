import numpy as np
from pathlib import Path
import h5py
from fenics import *
from fenics_ice import model
import argparse


def main(dd, infile, outfile, noise_sdev, L, seed=0, ls=None):
    """
    Take velocity data from run_momsolve.py and add gaussian noise

    dd - (string/path) directory where input is found & where output is written
    infile - (string/path) input file name (.h5)
    outfile - (string/path) output file name (.h5)
    noise_sdev - (float) standard deviation of added noise
    L - (float) Length of domain
    seed - (int) Random seed
    ls - (float [optional]) Spacing between points in a new grid

    Expects an HDF5 file (containing both mesh & velocity function) as input.
    In the case of a periodic boundary condition with NxN elements, this
    produces NxN velocity observations (NOT N+1 x N+1) because otherwise
    boundary nodes would be doubly constrained.

    If 'ls' is not provided, data will remain on the input grid.
    """

    assert Path(infile).suffix == ".h5"
    assert Path(outfile).suffix == ".h5"
    assert L > 0.0
    assert noise_sdev > 0.0

    infile = HDF5File(MPI.comm_world, str(Path(dd)/infile), 'r')

    # Get mesh from file
    mesh = Mesh()
    infile.read(mesh, 'mesh', False)
    periodic_bc = bool(infile.attributes('mesh')['periodic'])

    if periodic_bc:
        V = VectorFunctionSpace(mesh,
                                'Lagrange',
                                1,
                                dim=2,
                                constrained_domain=model.PeriodicBoundary(L))

    else:
        V = VectorFunctionSpace(mesh,
                                'Lagrange',
                                1,
                                dim=2)

    # Read the velocity
    U = Function(V)
    infile.read(U, 'U')

    if ls is not None:

        # Get the mesh coordinates (only to set bounds)
        mc = mesh.coordinates()
        xmin = mc[:, 0].min()
        xmax = mc[:, 0].max()

        ymin = mc[:, 1].min()
        ymax = mc[:, 1].max()

        # Generate ls-spaced points
        xc = np.arange(xmin + ls/2.0, xmax, ls)
        yc = np.arange(ymin + ls/2.0, ymax, ls)

        # Pretty hacky - turn these into a rectangular mesh
        # because it makes periodic interpolation easier
        pts_mesh = RectangleMesh(Point(xc[0], yc[0]),
                                 Point(xc[-1], yc[-1]),
                                 len(xc)-1, len(yc)-1)
        pts_space = VectorFunctionSpace(pts_mesh,
                                        'Lagrange',
                                        degree=1,
                                        dim=2)

        U_pts = project(U, pts_space)

    else:
        U_pts = U

    # How many points?
    ndofs = int(U_pts.vector()[:].shape[0] / 2)

    # Generate the random noise
    np.random.seed(seed)
    u_noise = np.random.normal(scale=noise_sdev, size=ndofs)
    v_noise = np.random.normal(scale=noise_sdev, size=ndofs)

    # Grab the two components of the velocity vector, and add noise
    U_vec = U_pts.vector()[:]
    U_vec[0::2] += u_noise
    U_vec[1::2] += v_noise

    u_array = U_vec[0::2]
    v_array = U_vec[1::2]

    # [::2] because tabulate_dof_coordinates produces two copies
    # (because 2 dofs per node...)
    x, y = np.hsplit(U_pts.function_space().tabulate_dof_coordinates()[::2], 2)

    # Produce output as raw points & vel
    output = h5py.File(Path(dd)/outfile, 'w')

    output.create_dataset("x",
                          x.shape,
                          dtype=x.dtype,
                          data=x)

    output.create_dataset("y",
                          x.shape,
                          dtype=x.dtype,
                          data=y)

    output.create_dataset("u_obs",
                          x.shape,
                          dtype=np.float64,
                          data=u_array)

    output.create_dataset("v_obs",
                          x.shape,
                          dtype=np.float64,
                          data=v_array)

    noise_arr = np.zeros_like(x)
    noise_arr[:] = noise_sdev

    output.create_dataset("u_std",
                           x.shape,
                           dtype=np.float64,
                           data=noise_arr)

    output.create_dataset("v_std",
                           x.shape,
                           dtype=np.float64,
                           data=noise_arr)

    mask_arr = np.ones_like(x)

    output.create_dataset("mask_vel",
                           x.shape,
                           dtype=np.float64,
                           data=mask_arr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', dest='dd', type=str,
                        required=True, help='Directory with input data')
    parser.add_argument('-i', '--infile', dest='infile', type=str,
                        required=True,
                        help='HDF5 File containing mesh & function')
    parser.add_argument('-o', '--outfile', dest='outfile', type=str,
                        required=True, help='Filename for HDF5 output')
    parser.add_argument('-s', '--sigma', dest='noise_sdev', type=float,
                        help='Standard deviation of added Gaussian Noise')
    parser.add_argument('-L', '--length', dest='L', type=int,
                        help='Length of IsmipC domain.')
    parser.add_argument('-l', '--ls', dest='ls', type=float,
                        help='Grid spacing for optional interpolation')
    parser.add_argument('-r', '--seed', dest='seed', type=int,
                        help='Random seed for noise generation')

    parser.set_defaults(noise_sdev=1.0, L=False, seed=0, ls=None)
    args = parser.parse_args()

    dd = args.dd
    infile = args.infile
    outfile = args.outfile
    noise_sdev = args.noise_sdev
    L = args.L
    seed = args.seed
    ls = args.ls
    main(dd, infile, outfile, noise_sdev, L, seed, ls)
