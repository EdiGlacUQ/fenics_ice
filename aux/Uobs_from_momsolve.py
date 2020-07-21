import numpy as np
from pathlib import Path
import h5py
from fenics import *
from fenics_ice import model
import argparse


def main(dd, infile, outfile, noise_sdev, L, seed=0):
    """
    Take velocity data from run_momsolve.py and add gaussian noise

    Expects an HDF5 file (containing both mesh & velocity function) as input.
    In the case of a periodic boundary condition with NxN elements, this
    produces NxN velocity observations (NOT N+1 x N+1) because otherwise
    boundary nodes would be doubly constrained.
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

    # How many points?
    ndofs = int(U.vector()[:].shape[0] / 2)

    # Generate the random noise
    np.random.seed(seed)
    u_noise = np.random.normal(scale=noise_sdev, size=ndofs)
    v_noise = np.random.normal(scale=noise_sdev, size=ndofs)

    # Grab the two components of the velocity vector, and add noise
    U_vec = U.vector()[:]
    U_vec[0::2] += u_noise
    U_vec[1::2] += v_noise

    u_array = U_vec[0::2]
    v_array = U_vec[1::2]

    # [::2] because tabulate_dof_coordinates produces two copies
    # (because 2 dofs per node...)
    x, y = np.hsplit(U.function_space().tabulate_dof_coordinates()[::2], 2)

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
    parser.add_argument('-r', '--seed', dest='seed', type=int,
                        help='Random seed for noise generation')

    parser.set_defaults(noise_sdev=1.0, L=False, seed=0)
    args = parser.parse_args()

    dd = args.dd
    infile = args.infile
    outfile = args.outfile
    noise_sdev = args.noise_sdev
    L = args.L
    seed = args.seed

    main(dd, infile, outfile, noise_sdev, L, seed)
