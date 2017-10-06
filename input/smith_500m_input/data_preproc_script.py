import sys
import numpy as np
import scipy.io as io
import scipy.interpolate as interp
import matplotlib.pyplot as plt

import numpy as np
from IPython import embed

############################
#Details of model domain grid
############################

xlim = [-1607000.0,-1383000.0]
ylim = [-717000.0,-523000.0]

Lx = xlim[1] - xlim[0]
Ly = ylim[1] - ylim[0]

nx = Lx/1e3
ny = Ly/1e3

###############
#BEDMAP2 data
##############
data_dir = '/Users/conradkoziol/Documents/Glaciology/Data/bedmap2_bin/'
tf = 'bedmap2_thickness.flt'
bf = 'bedmap2_bed.flt'
gsf = 'bedmap2_icemask_grounded_and_shelves.flt'

fid = open(data_dir + tf,"rb")
file_contents = np.fromfile(fid, dtype='float32')
bm_thick = np.reshape(file_contents, [6667,6667])

fid = open(data_dir + bf,"rb")
file_contents = np.fromfile(fid, dtype='float32')
bm_bed= np.reshape(file_contents, [6667,6667])

fid = open(data_dir + gsf,"rb")
file_contents = np.fromfile(fid, dtype='float32')
bm_shelves= np.reshape(file_contents, [6667,6667])

#Depress bed beneath shelves
bm_bed[bm_shelves==1] = 1.5*bm_bed[bm_shelves==1]

#Cell centre coords
bm_x = np.linspace(-3333000,3333000, 6667)
bm_y = np.linspace(3333000,-3333000, 6667)

#Create bufferd x,y masks
bf = 1e3
xm = (xlim[0] - bf < bm_x) & (bm_x < xlim[1] + bf)
ym = (ylim[0] - bf < bm_y) & (bm_y < ylim[1] + bf)

#Alternative to np.linspace which has depreciated support for floats
#xcoord = [xlim[0] + x*1e3 for x in np.arange(0,nx+1)]
#ycoord = [ylim[0] + x*1e3 for x in np.arange(0,ny+1)]
xcoord_bm = bm_x[xm]
ycoord_bm = bm_y[ym]

bed_ = bm_bed[ym,:];
bed = bed_[:,xm]

thick_ = bm_thick[ym,:];
thick = thick_[:,xm]
thick_vls = thick
thick[thick<1] = 0

shelves_ = bm_shelves[ym,:];
shelves = shelves_[:,xm]

mask = np.empty(thick.shape)
mask[thick_vls >= 1] = 1
mask[thick_vls < 1] = -10
mask[thick_vls == -9999] = 0

###############
#Measures data
#################

data_dir = '/Users/conradkoziol/Documents/Glaciology/Data/Measures_Antarctica/'
vf = 'measures450.mat'

mes_data = io.loadmat(data_dir + vf)
mes_uvel = mes_data['uvel']
mes_vvel = mes_data['vvel']
mes_x = np.squeeze(mes_data['xmeasures450'])
mes_y = np.squeeze(mes_data['ymeasures450'])

mes_uvel = np.flipud(mes_uvel)
mes_vvel = np.flipud(mes_vvel)
mes_y = np.flipud(mes_y)


#Create bufferd x,y masks
bf = 1e3
xm2 = (xlim[0] - bf < mes_x) & (mes_x < xlim[1] + bf)
ym2 = (ylim[0] - bf < mes_y) & (mes_y < ylim[1] + bf)

xcoord_ms = mes_x[xm2]
ycoord_ms = mes_y[ym2]

uvel_ = mes_uvel[ym2,:]
uvel = uvel_[:,xm2]

vvel_ = mes_vvel[ym2,:]
vvel = vvel_[:,xm2]

outfile = 'grid_data'
np.savez(outfile,nx=nx,ny=ny,xlim=xlim,ylim=ylim, Lx=Lx, Ly=Ly,
            xcoord_bm=xcoord_bm,ycoord_bm=ycoord_bm,
            xcoord_ms=xcoord_ms,ycoord_ms=ycoord_ms,
            bed=bed, thick=thick, mask=mask, uvel=uvel, vvel=vvel)


plt.figure()
plt.imshow(bed)
plt.title('Bed')
plt.savefig('bed.png')

plt.figure()
cax = plt.imshow(thick)
plt.title('Thickness >10m')
plt.colorbar(cax)
cax.set_clim(10, 2000)
plt.savefig('thick.png')

plt.figure()
cax = plt.imshow(thick < 10)
plt.title('Thickness <10m')
plt.savefig('thick2.png')

plt.figure()
plt.imshow(shelves==1)
plt.title('Ice Shelf')
plt.savefig('shelves.png')

plt.figure()
plt.imshow(mask)
plt.title('Mask')
plt.savefig('mask.png')

plt.figure()
plt.imshow((uvel**2.0 + vvel**2.0)**(1.0/2.0))
plt.title('Velocities')
plt.savefig('vel.png')
