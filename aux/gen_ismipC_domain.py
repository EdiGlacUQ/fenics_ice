from fenics import *
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
import test_domains
import fenics_util as fu


#Number of cells in grid
nx = 360;
ny = 360;

#Fenics mesh
L = 120e3
mesh = RectangleMesh(Point(0,0), Point(L, L), nx, ny)
V = FunctionSpace(mesh, 'DG',0)
v = Function(V)
n = V.dim()
d = mesh.geometry().dim()

dof_coordinates = V.tabulate_dof_coordinates()
dof_coordinates.resize((n, d))
dof_x = dof_coordinates[:, 0]
dof_y = dof_coordinates[:, 1]

#Sampling Mesh, identical to Fenics mesh
domain = test_domains.ismipC(L,nx=nx+1,ny=ny+1)
xcoord = domain.x
ycoord = domain.y
xycoord = (xcoord, ycoord)

#Data is not stored in an ordered manner on the fencis mesh.
#Using interpolation function to get correct grid ordering
bed_interp = interp.RegularGridInterpolator(xycoord, domain.bed)
height_interp = interp.RegularGridInterpolator(xycoord, domain.thick)
bmelt_interp = interp.RegularGridInterpolator(xycoord, domain.bmelt)
mask_interp = interp.RegularGridInterpolator(xycoord, domain.mask)
B2_interp = interp.RegularGridInterpolator(xycoord, domain.B2)


#Coordinates of DOFS of fenics mesh in order data is stored
dof_xy = (dof_x, dof_y)
bed = bed_interp(dof_xy)
height = height_interp(dof_xy)
bmelt = bmelt_interp(dof_xy)
mask = mask_interp(dof_xy)
B2 = B2_interp(dof_xy)


#Save mesh and data points at coordinates
dd = '../input/ismipC/'

File(''.join([dd,'ismipC_mesh.xml'])) << mesh

v.vector()[:] = bed.flatten()
File(''.join([dd,'ismipC_mesh_bed.xml'])) <<  v

v.vector()[:] = height.flatten()
File(''.join([dd,'ismipC_mesh_height.xml'])) <<  v

v.vector()[:] = bmelt.flatten()
File(''.join([dd,'ismipC_mesh_bmelt.xml'])) <<  v

v.vector()[:] = mask.flatten()
File(''.join([dd,'ismipC_mesh_mask.xml'])) <<  v

v.vector()[:] = B2.flatten()
File(''.join([dd,'ismipC_mesh_B2.xml'])) <<  v
