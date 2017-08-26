from fenics import *
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
import test_domains
import fenics_util as fu


#Number of cells in grid
nx = 150;
ny = 150;

#Fenics mesh
mesh = RectangleMesh(Point(0,0), Point(150e3, 150e3), nx, ny)
V = FunctionSpace(mesh, 'Lagrange',1)
v = Function(V)
n = V.dim()
d = mesh.geometry().dim()

dof_coordinates = V.tabulate_dof_coordinates()
dof_coordinates.resize((n, d))
dof_x = dof_coordinates[:, 0]
dof_y = dof_coordinates[:, 1]

#Sampling Mesh, identical to Fenics mesh
domain = test_domains.gldbg2013(nx=nx+1,ny=ny+1)
xcoord = domain.x
ycoord = domain.y

#Data is not stored in an ordered manner on the fencis mesh.
#Using interpolation function to get correct grid ordering
bed_interp = interp.RectBivariateSpline(xcoord,ycoord, domain.bed)
surf_interp = interp.RectBivariateSpline(xcoord,ycoord, domain.surf)
bmelt_interp = interp.RectBivariateSpline(xcoord,ycoord, domain.bmelt)
bdrag_interp = interp.RectBivariateSpline(xcoord,ycoord, domain.bdrag)

#Coordinates of DOFS of fenics mesh in order data is stored
bed = bed_interp.ev(dof_x, dof_y)
surf = surf_interp.ev(dof_x, dof_y)
bmelt = bmelt_interp.ev(dof_x, dof_y)
bdrag = bdrag_interp.ev(dof_x, dof_y)

#Save mesh and data points at coordinates
File('gldbg2013_mesh.xml') << mesh

v.vector()[:] = bed.flatten()
File('gldbg2013_mesh_bed.xml') <<  v

v.vector()[:] = surf.flatten()
File('gldbg2013_mesh_surf.xml') <<  v

v.vector()[:] = bmelt.flatten()
File('gldbg2013_mesh_bmelt.xml') <<  v

v.vector()[:] = bdrag.flatten()
File('gldbg2013_mesh_bdrag.xml') <<  v
