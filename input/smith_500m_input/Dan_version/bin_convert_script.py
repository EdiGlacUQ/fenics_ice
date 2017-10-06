import sys
sys.path.insert(0,'../../../code/')
from fenics import *
import scipy.interpolate as interp
import matplotlib.pyplot as plt

import numpy as np
import fenics_util as fu
from IPython import embed

#Details of data grid
nx = 499            #number of cells
ny = 431
xlim = [-1607500.0,-1382950.0]
ylim = [-717200.0,-523250.0]
Lx = xlim[1] - xlim[0]
Ly = ylim[1] - ylim[0]
xcoord = np.linspace(xlim[0],xlim[1],nx+1)
ycoord = np.linspace(ylim[0],ylim[1],ny+1)
outfile = 'grid_data'
np.savez(outfile,nx=nx,ny=ny,xlim=xlim,ylim=ylim)

#Load data
Hdata = fu.binread('HinitForward.bin')
Bdata = fu.binread('topogForward.bin')
Mdata = fu.binread('HmaskGavin.box')

#Vector -> Matrix
thick = np.reshape(Hdata, [ny+1, nx+1]); thick = thick.T
bed = np.reshape(Bdata, [ny+1, nx+1]); bed = bed.T
mask = np.reshape(Mdata, [ny+1, nx+1]); mask = mask.T
mask[mask == -1] = 0

plt.imshow(bed)
plt.savefig('bed.png')

plt.imshow(thick)
plt.savefig('thick.png')

#Fenics mesh
mesh = RectangleMesh(Point(xlim[0],ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny)
V = FunctionSpace(mesh, 'DG',0)
v = Function(V)
n = V.dim()
d = mesh.geometry().dim()

dof_coordinates = V.tabulate_dof_coordinates()
dof_coordinates.resize((n, d))
dof_x = dof_coordinates[:, 0]
dof_y = dof_coordinates[:, 1]

#Sampling Mesh, identical to Fenics mesh
xycoord = (xcoord, ycoord)


#Data is not stored in an ordered manner on the fencis mesh.
#Using interpolation function to get correct grid ordering
bed_interp = interp.RegularGridInterpolator(xycoord, bed)
thick_interp = interp.RegularGridInterpolator(xycoord, thick)
mask_interp = interp.RegularGridInterpolator(xycoord, mask, method='nearest')


#Coordinates of DOFS of fenics mesh in order data is stored
dof_xy = (dof_x, dof_y)
bed = bed_interp(dof_xy)
thick = thick_interp(dof_xy)
mask = mask_interp(dof_xy)


#Save mesh and data points at coordinates
dd = './'

File(''.join([dd,'smith450m_mesh.xml'])) << mesh

v.vector()[:] = bed.flatten()
File(''.join([dd,'smith450m_mesh_bed.xml'])) <<  v

v.vector()[:] = thick.flatten()
File(''.join([dd,'smith450m_mesh_thick.xml'])) <<  v

v.vector()[:] = mask.flatten()
File(''.join([dd,'smith450m_mesh_mask.xml'])) <<  v
