import os
import argparse
from pathlib import Path
from fenics import *
import scipy.interpolate as interp
import matplotlib.pyplot as plt

import numpy as np
import fenics_ice.fenics_util as fu

import h5py

def main(outdir):

    # outpath = "smith_input.h5"
    geom_outpath = "smith_geom.h5"
    vel_outpath = "smith_obs_vel.h5"
    smb_outpath = "smith_smb.h5"
    bglen_outpath = "smith_bglen.h5"
    mesh_outpath = "smith_mesh.xml"

    #Load preprocessed BEDMPA2 data by bedmap2_data_script.py
    fenics_base = Path(os.environ["FENICS_ICE_BASE_DIR"])
    infile = fenics_base / "input/smith_500m_input" / 'grid_data.npz'
    npzfile = np.load(infile)

    nx = int(npzfile['nx'])
    ny = int(npzfile['ny'])
    xlim = npzfile['xlim']
    ylim = npzfile['ylim']
    Lx = int(npzfile['Lx'])
    Ly = int(npzfile['Ly'])

    xcoord_bm = npzfile['xcoord_bm']   # 229 x 193
    ycoord_bm = npzfile['ycoord_bm']
    xcoord_ms = npzfile['xcoord_ms']   # 509 x 429
    ycoord_ms = npzfile['ycoord_ms']
    xcoord_di = npzfile['xcoord_di']   # 47 x 40
    ycoord_di = npzfile['ycoord_di']
    xcoord_smb = npzfile['xcoord_smb'] # 47 x 40
    ycoord_smb = npzfile['ycoord_smb']

    bed = npzfile['bed']     # 193 x 229
    thick = npzfile['thick'] # 193 x 229
    mask = npzfile['mask']   # 193 x 229

    uvel = npzfile['uvel']   # 429 x 509
    vvel = npzfile['vvel']   # 429 x 509
    ustd = npzfile['ustd']   # 429 x 509
    vstd = npzfile['vstd']   # 429 x 509
    mask_vel = npzfile['mask_vel']    # 429 x 509

    B_mod = npzfile['B']     # 40 x 47
    smb = npzfile['smb']     # 40 x 47

    outfile = h5py.File(outdir/geom_outpath, 'w')
    outfile.create_dataset("x",
                           xcoord_bm.shape,
                           dtype=xcoord_bm.dtype,
                           data=xcoord_bm)

    outfile.create_dataset("y",
                           ycoord_bm.shape,
                           dtype=ycoord_bm.dtype,
                           data=ycoord_bm)

    outfile.create_dataset("bed",
                           bed.shape,
                           dtype=bed.dtype,
                           data=bed)

    outfile.create_dataset("thick",
                           thick.shape,
                           dtype=thick.dtype,
                           data=thick)

    outfile.create_dataset("data_mask",
                           mask.shape,
                           dtype=mask.dtype,
                           data=mask)

    outfile.close()

    xx_vel, yy_vel = np.meshgrid(xcoord_ms, ycoord_ms)
    xx_vel = np.ravel(xx_vel)
    yy_vel = np.ravel(yy_vel)

    outfile = h5py.File(outdir/vel_outpath, 'w')

    outfile.create_dataset("x",
                           xx_vel.shape,
                           dtype=xx_vel.dtype,
                           data=xx_vel)

    outfile.create_dataset("y",
                           yy_vel.shape,
                           dtype=yy_vel.dtype,
                           data=yy_vel)

    outfile.create_dataset("u_obs",
                           xx_vel.shape,
                           dtype=uvel.dtype,
                           data=np.ravel(uvel))

    outfile.create_dataset("v_obs",
                           xx_vel.shape,
                           dtype=vvel.dtype,
                           data=np.ravel(vvel))

    outfile.create_dataset("u_std",
                           xx_vel.shape,
                           dtype=ustd.dtype,
                           data=np.ravel(ustd))

    outfile.create_dataset("v_std",
                           xx_vel.shape,
                           dtype=vstd.dtype,
                           data=np.ravel(vstd))

    outfile.create_dataset("mask_vel",
                           xx_vel.shape,
                           dtype=mask_vel.dtype,
                           data=np.ravel(mask_vel))

    outfile.close()

    outfile = h5py.File(outdir/smb_outpath, 'w')
    outfile.create_dataset("x",
                           xcoord_smb.shape,
                           dtype=xcoord_smb.dtype,
                           data=xcoord_smb)

    outfile.create_dataset("y",
                           ycoord_smb.shape,
                           dtype=ycoord_smb.dtype,
                           data=ycoord_smb)

    outfile.create_dataset("smb",
                           smb.shape,
                           dtype=smb.dtype,
                           data=smb)

    outfile.close()

    outfile = h5py.File(outdir/bglen_outpath, 'w')
    outfile.create_dataset("x",
                           xcoord_smb.shape,
                           dtype=xcoord_smb.dtype,
                           data=xcoord_smb)

    outfile.create_dataset("y",
                           ycoord_smb.shape,
                           dtype=ycoord_smb.dtype,
                           data=ycoord_smb)

    outfile.create_dataset("bglen",
                           B_mod.shape,
                           dtype=B_mod.dtype,
                           data=B_mod)

    outfile.close()

    mesh = RectangleMesh(Point(xlim[0], ylim[0]), Point(xlim[1], ylim[1]), nx, ny)

    File(str(outdir/mesh_outpath)) << mesh

    # #Fenics mesh
    # mesh = RectangleMesh(Point(xlim[0],ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny)
    # V = FunctionSpace(mesh, 'DG',0)
    # v = Function(V)
    # n = V.dim()
    # d = mesh.geometry().dim()

    # dof_coordinates = V.tabulate_dof_coordinates()
    # dof_coordinates.resize((n, d))
    # dof_x = dof_coordinates[:, 0]
    # dof_y = dof_coordinates[:, 1]

    # #Sampling Mesh, identical to Fenics mesh
    # xycoord_bm = (xcoord_bm, np.flipud(ycoord_bm))
    # xycoord_ms = (xcoord_ms, np.flipud(ycoord_ms))
    # xycoord_di = (xcoord_di, np.flipud(ycoord_di))
    # xycoord_smb = (xcoord_smb, np.flipud(ycoord_smb))


    # #Data is not stored in an ordered manner on the fencis mesh;
    # #using interpolation function to get correct grid ordering
    # #Note Transpose for x,y indexing
    # bed_interp = interp.RegularGridInterpolator(xycoord_bm, list(zip(*bed[::-1])))
    # thick_interp = interp.RegularGridInterpolator(xycoord_bm, list(zip(*thick[::-1])))
    # mask_interp = interp.RegularGridInterpolator(xycoord_bm, list(zip(*mask[::-1])), method='nearest')

    # mask_bin = np.isclose(mask,1.0).astype(int)   #linear/nearest for edge thickness correction
    # maskl_interp = interp.RegularGridInterpolator(xycoord_bm, list(zip(*mask_bin[::-1])))
    # maskn_interp = interp.RegularGridInterpolator(xycoord_bm, list(zip(*mask_bin[::-1])), method='nearest')

    # uvel_interp = interp.RegularGridInterpolator(xycoord_ms, list(zip(*uvel[::-1])))
    # vvel_interp = interp.RegularGridInterpolator(xycoord_ms, list(zip(*vvel[::-1])))
    # ustd_interp = interp.RegularGridInterpolator(xycoord_ms, list(zip(*ustd[::-1])))
    # vstd_interp = interp.RegularGridInterpolator(xycoord_ms, list(zip(*vstd[::-1])))
    # mask_vel_interp = interp.RegularGridInterpolator(xycoord_ms, list(zip(*mask_vel[::-1])), method='nearest')

    # B_interp = interp.RegularGridInterpolator(xycoord_di, list(zip(*B_mod[::-1])))
    # smb_interp = interp.RegularGridInterpolator(xycoord_smb, list(zip(*smb[::-1])))

    # #Coordinates of DOFS of fenics mesh in order data is stored
    # dof_xy = (dof_x, dof_y)
    # mask = mask_interp(dof_xy)
    # maskl = maskl_interp(dof_xy)
    # maskn = maskn_interp(dof_xy)
    # bed = bed_interp(dof_xy)
    # thick_ = thick_interp(dof_xy)
    # thick = np.array([0.0 if np.isclose(mn,0) else t/ml for ml,mn,t in zip(maskl,maskn,thick_)])

    # u_obs = uvel_interp(dof_xy)
    # v_obs = vvel_interp(dof_xy)
    # u_std = ustd_interp(dof_xy)
    # v_std = vstd_interp(dof_xy)
    # mask_vel_ = mask_vel_interp(dof_xy)
    # mask_vel = np.logical_and(mask, mask_vel_ )

    # B_ = B_interp(dof_xy)
    # B_mod = np.array([0.0 if np.isclose(mn,0) else b/ml for ml,mn,b in zip(maskl,maskn,B_)])

    # smb = smb_interp(dof_xy)

    # from IPython import embed; embed()


    # #Save mesh and data points at coordinates
    # dd = './'

    # File(''.join([dd,'mesh.xml'])) << mesh

    # v.vector()[:] = bed.flatten()
    # File(''.join([dd,'bed.xml'])) <<  v

    # v.vector()[:] = thick.flatten()
    # File(''.join([dd,'thick.xml'])) <<  v

    # v.vector()[:] = mask.flatten()
    # File(''.join([dd,'mask.xml'])) <<  v

    # v.vector()[:] = u_obs.flatten()
    # File(''.join([dd,'u_obs.xml'])) <<  v

    # v.vector()[:] = v_obs.flatten()
    # File(''.join([dd,'v_obs.xml'])) <<  v

    # v.vector()[:] = u_std.flatten()
    # File(''.join([dd,'u_std.xml'])) <<  v

    # v.vector()[:] = v_std.flatten()
    # File(''.join([dd,'v_std.xml'])) <<  v

    # v.vector()[:] = mask_vel.flatten()
    # File(''.join([dd,'mask_vel.xml'])) <<  v

    # v.vector()[:] = B_mod.flatten()
    # File(''.join([dd,'Bglen.xml'])) <<  v

    # v.vector()[:] = smb.flatten()
    # File(''.join([dd,'smb.xml'])) <<  v

    # v.vector()[:] = 0.0
    # File(''.join([dd,'bmelt.xml'])) <<  v


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', dest='outdir', type=str,
                        help='Directory to store the output')
    parser.set_defaults(outdir='.')
    args = parser.parse_args()

    main(Path(args.outdir))
