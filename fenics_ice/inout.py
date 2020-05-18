"""
Module to handle model input & output
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from fenics import *
import logging

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


# Custom formatter
class LogFormatter(logging.Formatter):

    critical_fmt = "CRITICAL ERROR: %(msg)s"
    err_fmt  = "ERROR: %(msg)s"
    warning_fmt  = "WARNING: %(msg)s"
    dbg_fmt  = "DBG: %(module)s: Line %(lineno)d: %(msg)s"
    info_fmt = "%(msg)s"

    def __init__(self):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = self.dbg_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_fmt

        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warning_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = self.err_fmt

        elif record.levelno == logging.CRITICAL:
            self._style._fmt = self.critical_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result



def setup_logging(params):
    """
    Set up logging to file specified in params
    """

    log_level = params.io.log_level

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    fmt = LogFormatter()
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(fmt)

    logging.root.addHandler(hdlr)
    logging.root.setLevel(numeric_level)
    #Consider adding %(process)d- when we move to parallel sims (at least for debug messages?)
#    logging.basicConfig(level=numeric_level, format='%(levelname)s:%(message)s')
#    logging.basicConfig(level=numeric_level, format=)

    #e.g.:
    # logging.critical("critical")
    # logging.error("error")
    # logging.warning("warning")
    # logging.info("info")
    # logging.debug("boring")
