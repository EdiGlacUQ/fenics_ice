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
Generate ismipC domain in HDF5 grid format

Note that this produces arrays which are arranged [y,x]
"""

import os
import argparse
import h5py
from pathlib import Path

from fenics_ice import test_domains

def main(outfname, L, nx, ny, reflect):

    # Check valid filename & create dir if necessary
    outpath = Path(outfname)
    assert(outpath.suffix == ".h5")
    outdir = outpath.parent
    outdir.mkdir(exist_ok=True, parents=True)

    # Data on grid
    domain = test_domains.ismipC(L, nx=nx+1, ny=ny+1, tiles=1.0, reflect=reflect)

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

    parser.add_argument('-r', '--reflect', action='store_true', help='Produce a reflected/transposed ismipc domain (backwards compatibility)')

    parser.set_defaults(outfname='../input/ismipC/ismipC_input.h5',
                        L=40e3,
                        nx=100,
                        ny=100,
                        reflect=False)

    args = parser.parse_args()

    main(args.outfname, args.L, args.nx, args.ny, args.reflect)
