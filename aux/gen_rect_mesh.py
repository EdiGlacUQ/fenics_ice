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
Generate a rectangular mesh for (synthetic) fenics_ice sims. Previously, run_inv.py was
responsible for generating the mesh at run time.
"""

from mpi4py import MPI
import argparse
from fenics import RectangleMesh, Point, File

def gen_rect_mesh(nx, ny, xmin, xmax, ymin, ymax, outfile, direction='right'):

    mesh = RectangleMesh(MPI.COMM_SELF,
                         Point(xmin, ymin),
                         Point(xmax, ymax),
                         nx, ny, direction)

    File(MPI.COMM_SELF, outfile) << mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nx', '--nx', dest='nx', type=int,
                        help='Number of cells along x-axis direction')
    parser.add_argument('-ny', '--ny', dest='ny', type=int,
                        help='Number of cells along y-axis direction')
    parser.add_argument('-xmin', '--xmin', dest='xmin', type=float,
                        help='Min x coordinate')
    parser.add_argument('-xmax', '--xmax', dest='xmax', type=float,
                        help='Max x coordinate')
    parser.add_argument('-ymin', '--ymin', dest='ymin', type=float,
                        help='Min y coordinate')
    parser.add_argument('-ymax', '--ymax', dest='ymax', type=float,
                        help='Max y coordinate')
    parser.add_argument('-o', '--outfile', dest='outfile', type=str,
                        help='Mesh output filename (eg.xml)')
    parser.add_argument('-d', '--direction', dest='direction', type=str,
                        help="Left, Right or Crossed diagonals")
    parser.set_defaults(outfile="mesh.xml", xmin=0, ymin=0, direction='right')

    args = parser.parse_args()

    gen_rect_mesh(args.nx, args.ny, args.xmin,
         args.xmax, args.ymin, args.ymax,
         args.outfile, args.direction)
