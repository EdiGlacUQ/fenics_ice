


import mds
import sys
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt


def quickplot(tstep):

  #Acquire Data
  fn = 'land_ice'
  file_data = mds.rdmds(fn, tstep)

  Uvel = file_data[0,:,];
  Vvel = file_data[1,:,];
  Vel = np.sqrt(np.square(Uvel) + np.square(Vvel))
  thick = file_data[2,:,];

  plt.figure(1)
  plt.imshow(Uvel)
  plt.colorbar()
  plt.title('U velocities')

  plt.figure(2)
  plt.imshow(Vvel)
  plt.colorbar()
  plt.title('V velocities')
  
  plt.figure(3)
  plt.imshow(Vel)
  plt.colorbar()
  plt.title('Velocities')

  plt.figure(4)
  plt.imshow(thick)
  plt.colorbar()
  plt.title('Thickness')
  
  plt.show()
  
  return file_data
  
def siout(tsteps):
  fn = 'land_ice'
  file_data = mds.rdmds(fn, tsteps)
  return file_data

def conv_vel(tsteps):
  fn = 'land_ice'
  file_data = mds.rdmds(fn, tsteps)
  
  num_tsteps = len(tsteps)
  vel_diff = np.zeros(num_tsteps)
  v_curr = 0;
  v_prev = 0;
  for i in range(num_tsteps):  
    Uvel = file_data[i,0,:,];
    Vvel = file_data[i,1,:,];
    Vel = np.sqrt(np.square(Uvel) + np.square(Vvel)) 
    v_prev = v_curr;
    v_curr = Vel
    vel_diff[i] = np.linalg.norm(v_curr-v_prev)

  plt.figure()
  plt.plot(tsteps[1:],vel_diff[1:])
  plt.show()
  
  return vel_diff 

def conv_height(tsteps):
  fn = 'land_ice'
  file_data = mds.rdmds(fn, tsteps)
  
  num_tsteps = len(tsteps)
  height_diff = np.zeros(num_tsteps)
  h_curr = 0;
  h_prev = 0;
  for i in range(num_tsteps):  
    height = file_data[i,2,:,];
    h_prev = h_curr;
    h_curr = height
    height_diff[i] = np.linalg.norm(h_curr-h_prev)

  plt.figure()
  plt.plot(tsteps[1:],height_diff[1:])
  plt.yscale('log')
  plt.show()
  
  return height_diff  
  


def binread(fn):
  fid = open(fn,"rb")
  file_contents = np.fromfile(fn, dtype='float64')
  if sys.byteorder == 'little': file_contents.byteswap(True)
  fid.close()
  return file_contents


def writefield(fname,data):
    print 'write to file: '+fname
    if sys.byteorder == 'little': data.byteswap(True)
    fid = open(fname,"wb")
    data.tofile(fid)
    fid.close()


def output2bin(tstep,fld,fname):

  #Acquire Data
  fn = 'land_ice'
  file_data = mds.rdmds(fn, tstep)
  fld_data = file_data[fld,:,];
  
  writefield(fname,fld_data)

