import sys
sys.path.insert(0,'../../code/')
from fenics import *
import scipy.interpolate as interp
import matplotlib.pyplot as plt

import numpy as np
import fenics_util as fu
from IPython import embed

#Load preprocessed BEDMPA2 data by bedmap2_data_script.py
infile = 'grid_data.npz'
npzfile = np.load(infile)

nx = int(npzfile['nx'])
ny = int(npzfile['ny'])
xlim = npzfile['xlim']
ylim = npzfile['ylim']
Lx = int(npzfile['Lx'])
Ly = int(npzfile['Ly'])
xcoord_bm = npzfile['xcoord_bm']
ycoord_bm = npzfile['ycoord_bm']
xcoord_ms = npzfile['xcoord_ms']
ycoord_ms = npzfile['ycoord_ms']
xcoord_di = npzfile['xcoord_di']
ycoord_di = npzfile['ycoord_di']
bed = npzfile['bed']
thick = npzfile['thick']
mask = npzfile['mask']
uvel = npzfile['uvel']
vvel = npzfile['vvel']
ustd = npzfile['ustd']
vstd = npzfile['vstd']
mask_vel = npzfile['mask_vel']
B_mod = npzfile['B']

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
xycoord_bm = (xcoord_bm, np.flipud(ycoord_bm))
xycoord_ms = (xcoord_ms, np.flipud(ycoord_ms))
xycoord_di = (xcoord_di, np.flipud(ycoord_di))

#Data is not stored in an ordered manner on the fencis mesh;
#using interpolation function to get correct grid ordering
#Note Transpose for x,y indexing
bed_interp = interp.RegularGridInterpolator(xycoord_bm, zip(*bed[::-1]))
thick_interp = interp.RegularGridInterpolator(xycoord_bm, zip(*thick[::-1]))
mask_interp = interp.RegularGridInterpolator(xycoord_bm, zip(*mask[::-1]), method='nearest')

mask_bin = np.isclose(mask,1.0).astype(int)   #linear/nearest for edge thickness correction
maskl_interp = interp.RegularGridInterpolator(xycoord_bm, zip(*mask_bin[::-1]))
maskn_interp = interp.RegularGridInterpolator(xycoord_bm, zip(*mask_bin[::-1]), method='nearest')

uvel_interp = interp.RegularGridInterpolator(xycoord_ms, zip(*uvel[::-1]))
vvel_interp = interp.RegularGridInterpolator(xycoord_ms, zip(*vvel[::-1]))
ustd_interp = interp.RegularGridInterpolator(xycoord_ms, zip(*ustd[::-1]))
vstd_interp = interp.RegularGridInterpolator(xycoord_ms, zip(*vstd[::-1]))
mask_vel_interp = interp.RegularGridInterpolator(xycoord_ms, zip(*mask_vel[::-1]), method='nearest')

B_interp = interp.RegularGridInterpolator(xycoord_di, zip(*B_mod[::-1]))

#Coordinates of DOFS of fenics mesh in order data is stored
dof_xy = (dof_x, dof_y)
mask = mask_interp(dof_xy)
maskl = maskl_interp(dof_xy)
maskn = maskn_interp(dof_xy)
bed = bed_interp(dof_xy)
thick_ = thick_interp(dof_xy)
thick = np.array([0.0 if np.isclose(mn,0) else t/ml for ml,mn,t in zip(maskl,maskn,thick_)])

u_obs = uvel_interp(dof_xy)
v_obs = vvel_interp(dof_xy)
u_std = ustd_interp(dof_xy)
v_std = vstd_interp(dof_xy)
mask_vel_ = mask_vel_interp(dof_xy)
mask_vel = np.logical_and(mask, mask_vel_ )

B_ = B_interp(dof_xy)
B_mod = np.array([0.0 if np.isclose(mn,0) else b/ml for ml,mn,b in zip(maskl,maskn,B_)])


#Save mesh and data points at coordinates
dd = './'

File(''.join([dd,'smith450m_mesh.xml'])) << mesh

v.vector()[:] = bed.flatten()
File(''.join([dd,'smith450m_mesh_bed.xml'])) <<  v

v.vector()[:] = thick.flatten()
File(''.join([dd,'smith450m_mesh_thick.xml'])) <<  v

v.vector()[:] = mask.flatten()
File(''.join([dd,'smith450m_mesh_mask.xml'])) <<  v

v.vector()[:] = u_obs.flatten()
File(''.join([dd,'smith450m_mesh_u_obs.xml'])) <<  v

v.vector()[:] = v_obs.flatten()
File(''.join([dd,'smith450m_mesh_v_obs.xml'])) <<  v

v.vector()[:] = u_std.flatten()
File(''.join([dd,'smith450m_mesh_u_std.xml'])) <<  v

v.vector()[:] = v_std.flatten()
File(''.join([dd,'smith450m_mesh_v_std.xml'])) <<  v

v.vector()[:] = mask_vel.flatten()
File(''.join([dd,'smith450m_mesh_mask_vel.xml'])) <<  v

v.vector()[:] = B_mod.flatten()
File(''.join([dd,'smith450m_mesh_mask_B_mod.xml'])) <<  v
