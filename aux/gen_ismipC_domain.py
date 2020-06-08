"""Generate ismipC domain in HDF5 grid format"""

import os
import argparse
import h5py
from pathlib import Path

from fenics import File, RectangleMesh, Point
from fenics_ice import test_domains

def main(outfname, L, nx, ny):

    # Check valid filename & create dir if necessary
    outpath = Path(outfname)
    assert(outpath.suffix == ".h5")
    outdir = outpath.parent
    outdir.mkdir(exist_ok=True, parents=True)

    mesh = RectangleMesh(Point(0,0), Point(L, L), nx, ny)
    File(str(outdir/'momsolve_mesh.xml')) << mesh

    # Data on grid
    domain = test_domains.ismipC(L, nx=nx+1, ny=ny+1, tiles=1.0)

    # Dictionary linking variable names to domain objects
    var_dict = {
        "bed": domain.bed,
        "thick": domain.thick,
        "data_mask": domain.mask,
        "bmelt": domain.bmelt,
        "smb": domain.smb,
        "B2": domain.B2,
        "alpha": domain.B2**0.5,
        "Bglen": domain.Bglen
    }

    outfile = h5py.File(outpath, 'w')

    # Metadata
    outfile.attrs['nx'] = nx+1
    outfile.attrs['ny'] = ny+1
    outfile.attrs['xmin'] = 0
    outfile.attrs['ymin'] = 0
    outfile.attrs['xmax'] = L
    outfile.attrs['ymax'] = L

    outfile.create_dataset("x",
                           domain.x.shape,
                           dtype=domain.x.dtype,
                           data=domain.x)

    outfile.create_dataset("y",
                           domain.y.shape,
                           dtype=domain.y.dtype,
                           data=domain.y)

    for key in var_dict.keys():
        outfile.create_dataset(key,
                               var_dict[key].shape,
                               dtype=var_dict[key].dtype,
                               data=var_dict[key])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-L', '--length', dest='L', type=int, help='Length of IsmipC domain.')
    parser.add_argument('-nx', '--nx', dest='nx', type=int, help='Number of cells along x-axis direction')
    parser.add_argument('-ny', '--ny', dest='ny', type=int, help='Number of cells along y-axis direction')

    parser.add_argument('-o', '--outfile', dest='outfname', type=str, help='Filename to store the output')

    parser.set_defaults(outfname='../input/ismipC/ismipC_input.h5',
                        L=40e3,
                        nx=100,
                        ny=100)

    args = parser.parse_args()

    main(args.outfname, args.L, args.nx, args.ny)
