import numpy as np
import sys
import os
from matplotlib import colors
from pylab import plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fenics import *

def plot_variable(u, name, direc, cmap='gist_yarg', scale='lin', numLvls=12,
                  umin=None, umax=None, tp=False, tpAlpha=0.5, show=True,
                  hide_ax_tick_labels=False, label_axes=True, title='',
                  use_colorbar=True, hide_axis=False, colorbar_loc='right'):
  """
  """
  mesh = u.function_space().mesh()
  v    = u.compute_vertex_values(mesh)
  x    = mesh.coordinates()[:,0]
  y    = mesh.coordinates()[:,1]
  t    = mesh.cells()

  d    = os.path.dirname(direc)
  if not os.path.exists(d):
    os.makedirs(d)

  if umin != None:
    vmin = umin
  else:
    vmin = v.min()
  if umax != None:
    vmax = umax
  else:
    vmax = v.max()

  # countour levels :
  if scale == 'log':
    v[v < vmin] = vmin + 1e-12
    v[v > vmax] = vmax - 1e-12
    from matplotlib.ticker import LogFormatter
    levels      = np.logspace(np.log10(vmin), np.log10(vmax), numLvls)
    formatter   = LogFormatter(10, labelOnlyBase=False)
    norm        = colors.LogNorm()

  elif scale == 'lin':
    v[v < vmin] = vmin + 1e-12
    v[v > vmax] = vmax - 1e-12
    from matplotlib.ticker import ScalarFormatter
    levels    = np.linspace(vmin, vmax, numLvls)
    formatter = ScalarFormatter()
    norm      = None

  elif scale == 'bool':
    from matplotlib.ticker import ScalarFormatter
    levels    = [0, 1, 2]
    formatter = ScalarFormatter()
    norm      = None

  fig = plt.figure(figsize=(5,5))
  ax  = fig.add_subplot(111)

  c = ax.tricontourf(x, y, t, v, levels=levels, norm=norm,
                     cmap=plt.get_cmap(cmap))
  plt.axis('equal')

  if tp == True:
    p = ax.triplot(x, y, t, '-', lw=0.2, alpha=tpAlpha)
  ax.set_xlim([x.min(), x.max()])
  ax.set_ylim([y.min(), y.max()])
  if label_axes:
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
  if hide_ax_tick_labels:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
  if hide_axis:
    plt.axis('off')

  # include colorbar :
  if scale != 'bool' and use_colorbar:
    divider = make_axes_locatable(plt.gca())
    cax  = divider.append_axes(colorbar_loc, "5%", pad="3%")
    cbar = plt.colorbar(c, cax=cax, format=formatter,
                        ticks=levels)
    tit = plt.title(title)

  if use_colorbar:
    plt.tight_layout(rect=[.03,.03,0.97,0.97])
  else:
    plt.tight_layout()
  plt.savefig(direc + name + '.png', dpi=300)
  if show:
    plt.show()
  plt.close(fig)

def plot_inv_conv(fvals, name, direc):
    plt.figure()
    plt.semilogy(fvals, 'ko-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.savefig(direc + name + '.png', dpi=300)
    plt.close()


def binread(fn):
  fid = open(fn,"rb")
  file_contents = np.fromfile(fn, dtype='float64')
  if sys.byteorder == 'little': file_contents.byteswap(True)
  fid.close()
  return file_contents

def conjgrad(A, b, x0 = None, max_iter = np.Inf, tol=1e-10, verbose =True):

    fs = b.function_space()
    x, p, r = Function(fs), Function(fs), Function(fs)

    x.assign(Constant(0.0) if x0 is None else x0)
    p.assign(Constant(0.0))
    r.assign(Constant(0.0))

    max_iter = min(b.vector().size(), max_iter)


    r.assign(b - A(x))
    p.assign(r)
    rs = innerprod(r, r)

    if np.sqrt(rs) < tol:
        print '''Norm of residual is less than tolerance (%1.2e) \n
                Converged in %i iterations ''' % (tol, 0)
        return x


    for i in range(max_iter):
        Ap = A(p)
        print 'Inner Product p * Ap %i' % innerprod(p, Ap)
        alpha = rs / innerprod(p, Ap)
        axpy(x, alpha, p)
        axpy(r, -alpha, Ap)
        rs_new = innerprod(r, r)

        #Check stopping criterion
        if np.sqrt(rs_new) < tol:
            print '''Norm of residual is less than tolerance (%1.2e)
            Converged in %i iterations ''' % (tol, i)
            return x

        elif i==max_iter:
            print '''Maximum number of iterations reached (%i)
            Norm of residual is (%1.2e)''' % (max_iter, np.sqrt(rs_new))
            return x

        p.assign(r + (rs_new / rs) * p)
        rs = rs_new

        if verbose:
            print 'Iteration: %i, Norm of residual: %1.2e' % (i,np.sqrt(rs_new))

def innerprod(x, y):
    inner = 0.0
    x = x.vector()
    y = y.vector()
    inner += x.inner(y)
    return inner

def axpy(x, alpha, y):
    x = x.vector()
    y = y.vector()
    x.axpy(alpha, y)
