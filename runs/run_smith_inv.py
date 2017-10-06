import sys
sys.path.insert(0,'../code/')
from fenics import *
import model
import solver
import matplotlib.pyplot as plt
import numpy as np
import fenics_util as fu
import time
from IPython import embed


set_log_level(20)

#Load Data
dd = '../input/smith_500m_input/'
data_mesh = Mesh(''.join([dd,'smith450m_mesh.xml']))
Q = FunctionSpace(data_mesh, 'DG', 0)
bed = Function(Q,''.join([dd,'smith450m_mesh_bed.xml']))
thick = Function(Q,''.join([dd,'smith450m_mesh_thick.xml']))
mask = Function(Q,''.join([dd,'smith450m_mesh_mask.xml']))
u_obs = Function(Q,''.join([dd,'smith450m_mesh_u_obs.xml']))
v_obs = Function(Q,''.join([dd,'smith450m_mesh_v_obs.xml']))


#Generate model mesh
gf = 'grid_data.npz'
npzfile = np.load(''.join([dd,'grid_data.npz']))
nx = int(npzfile['nx'])
ny = int(npzfile['ny'])
xlim = npzfile['xlim']
ylim = npzfile['ylim']

mesh = RectangleMesh(Point(xlim[0],ylim[0]), Point(xlim[-1], ylim[-1]), nx, ny)

#Initialize Model
param = {'eq_def' : 'action',
        'solver': 'default',
        'outdir' :'./output_smith_inv/'}
mdl = model.model(mesh,param)
mdl.init_bed(bed)
mdl.init_thick(thick)
mdl.init_mask(mask)
mdl.init_vel_obs(u_obs,v_obs)
#mdl.gen_ice_mask()
mdl.init_bmelt(Constant(0.0))
mdl.init_alpha(Constant(ln(2000)))
mdl.gen_domain()

#Solve
slvr = solver.ssa_solver(mdl)
slvr.def_mom_eq()
slvr.solve_mom_eq()


embed()

#Inversions
slvr = solver.ssa_solver(mdl)
slvr.inversion()

embed()

#Plot
alpha_inv = project(exp(slvr.alpha_inv),mdl.Q)
F_vals = [x for x in slvr.F_vals if x > 0]

fu.plot_variable(alpha_inv, 'alpha_inverted', mdl.param['outdir'])
fu.plot_inv_conv(F_vals, 'convergence', mdl.param['outdir'])


outdir = mdl.param['outdir']

vtkfile = File(''.join([outdir,'U.pvd']))
U = project(mdl.U,mdl.V)
vtkfile << U

vtkfile = File(''.join([outdir,'bed.pvd']))
vtkfile << mdl.bed

vtkfile = File(''.join([outdir,'thick.pvd']))
vtkfile << mdl.thick

vtkfile = File(''.join([outdir,'mask.pvd']))
vtkfile << mdl.mask

vtkfile = File(''.join([outdir,'uvel.pvd']))
vtkfile << mdl.u_obs

vtkfile = File(''.join([outdir,'vvel.pvd']))
vtkfile << mdl.v_obs
