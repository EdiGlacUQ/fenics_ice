from fenics import *
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
import test_domains
import os


#Number of cells in grid
nx = 100;
ny = 100;

#Length of domain
L = 40e3

#Output Location
dd = '../input/ismipC/'

#Fenics mesh
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
domain = test_domains.ismipC(L,nx=nx+1,ny=ny+1, tiles=1.0)
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
Bglen_interp = interp.RegularGridInterpolator(xycoord, domain.Bglen)

#Coordinates of DOFS of fenics mesh in order data is stored
dof_xy = (dof_x, dof_y)
bed = bed_interp(dof_xy)
height = height_interp(dof_xy)
bmelt = bmelt_interp(dof_xy)
mask = mask_interp(dof_xy)
B2 = B2_interp(dof_xy)
Bglen = Bglen_interp(dof_xy)



outfile = 'grid_data'
np.savez(os.path.join(dd,outfile),nx=nx,ny=ny,xlim=[0,L],ylim=[0,L], Lx=L, Ly=L)

File(os.path.join(dd,'mesh.xml')) << mesh

v.vector()[:] = bed.flatten()
File(os.path.join(dd,'bed.xml')) <<  v

v.vector()[:] = height.flatten()
File(os.path.join(dd,'thick.xml')) <<  v

v.vector()[:] = mask.flatten()
File(os.path.join(dd,'mask.xml')) <<  v

v.vector()[:] = bmelt.flatten()
File(os.path.join(dd,'bmelt.xml')) <<  v

v.vector()[:] = B2.flatten()
File(os.path.join(dd,'B2.xml')) <<  v

v.vector()[:] = Bglen.flatten()
File(os.path.join(dd,'Bglen.xml')) <<  v
