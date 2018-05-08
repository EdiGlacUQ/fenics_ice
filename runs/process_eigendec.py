import os
import sys
import argparse
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from IPython import embed


def main(inputfile, outdir, dd):

    pd = pickle.load(open(inputfile))
    lam = pd[0]
    eigenvecs = pd[1]

    lpos = np.argwhere(lam > 0)
    lneg = np.argwhere(lam < 0)
    lind = np.arange(0,len(lam))
    embed()

    plt.figure()
    plt.semilogy(lind[lpos], lam[lpos], 'b.')
    plt.semilogy(lind[lneg], np.abs(lam[lneg]), 'r.')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Value')
    plt.legend(['Positive','Negative eigenvalues'])
    plt.savefig(os.path.join(outdir, 'eigplot.png'))

    mesh = Mesh(os.path.join(dd, 'mesh.xml'))
    M = FunctionSpace(mesh, 'Lagrange', 1)
    efunc = Function(M)

    vtkfile = File(os.path.join(outdir,'eigenfuncs.pvd'))
    for ev in eigenvecs.T:
        efunc.vector().set_local(ev)
        efunc.vector().apply('insert')
        vtkfile << efunc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, help='Directory to store output')
    parser.add_argument('-d', '--datadir', dest='dd', type=str, required=True, help='Input file to process')
    parser.add_argument('-i', '--inputfile', dest='inputfile', type=str, required=True, help='Input file to process')

    parser.set_defaults(outdir ='./eigendec')
    args = parser.parse_args()

    outdir = args.outdir
    dd = args.dd
    inputfile = args.inputfile


    if not os.path.isfile(inputfile):
        print('File does not exists')
        sys.exit()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    main(inputfile, outdir, dd)
