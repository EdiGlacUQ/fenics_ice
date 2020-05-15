"""
Module to handle model input & output
"""

from pathlib import Path
import pickle
import numpy as np
from fenics import *

def write_qval(Qval, params):
    """
    Produces pickle dump with QOI value through time
    """

    outdir = params.io.output_dir
    filename = params.io.qoi_file

    run_length = params.time.run_length
    n_steps = params.time.total_steps
    ts = np.linspace(0, run_length, n_steps+1)

    pickle.dump([Qval, ts], (Path(outdir)/filename).open('wb'))

def write_dqval(dQ_ts, params):
    """
    Produces .pvd & .h5 files with dQoi_dCntrl
    """

    outdir = params.io.output_dir
    vtk_filename = params.io.dqoi_vtkfile
    h5_filename = params.io.dqoi_h5file

    vtkfile = File(str(Path(outdir)/vtk_filename))
    hdf5out = HDF5File(MPI.comm_world, str(Path(outdir)/h5_filename), 'w')
    n = 0.0

    for j in dQ_ts:
        #TODO - if we generalise cntrl in run_forward.py to be always a list
        #(possible dual inversion), should change this.
        # assert len(j) == 1, "Not yet implemented for dual inversion"
        # output = j[0]
        output = j
        output.rename('dQ', 'dQ')
        vtkfile << output
        hdf5out.write(output, 'dQ', n)
        n += 1.0

    hdf5out.close()
