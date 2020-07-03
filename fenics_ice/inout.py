"""
Module to handle model input & output
"""

import sys
from pathlib import Path
import pickle
import logging
import re
import h5py
import git
from scipy import interpolate as interp
from IPython import embed

from fenics import *
import numpy as np

# Regex for catching unnamed vars
unnamed_re = re.compile("f_[0-9]+")

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
    h5_filename = params.io.dqoi_h5file

    vtkfile = File(str((Path(outdir)/h5_filename).with_suffix(".pvd")))

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

def write_variable(var, params, name=None):
    """
    Produce xml & vtk output of supplied variable (prefixed with run name)
    Name is taken from variable structure if not provided
    """

    #Take variable's inbuilt name by default
    if not name:
        name = var.name()
        if unnamed_re.match(name):
            logging.error("Attempted to write out an unnamed variable %s" % name)
            raise Exception

    #Prefix the run name
    outfname = Path(params.io.output_dir)/"_".join((params.io.run_name,name))
    vtk_fname = str(outfname.with_suffix(".pvd"))
    xml_fname = str(outfname.with_suffix(".xml"))

    File(vtk_fname) << var
    File(xml_fname) << var

    logging.info("Writing function %s to file %s" % (name, outfname))

def read_vel_obs(params, model=None):
    """
    Read velocity observations & uncertainty from HDF5 file

    For now, expects an HDF5 file
    """

    infile = Path(params.io.input_dir) / params.obs.vel_file
    assert infile.exists(), f"Couldn't find velocity observations file: {infile}"

    infile = h5py.File(infile, 'r')

    x_obs = infile['x'][:, 0]
    y_obs = infile['y'][:, 0]
    u_obs = infile['u_obs'][:, 0]
    v_obs = infile['v_obs'][:, 0]
    u_std = infile['u_std'][:, 0]
    v_std = infile['v_std'][:, 0]
    mask_vel = infile['mask_vel'][:, 0]

    uv_obs_pts = np.vstack((x_obs, y_obs)).T

    if model is not None:
        model.uv_obs_pts = uv_obs_pts
        model.u_obs = u_obs
        model.v_obs = v_obs
        model.u_std = u_std
        model.v_std = v_std
        model.mask_vel = mask_vel
    else:
        return uv_obs_pts, u_obs, v_obs, u_std, v_std, mask_vel

class DataNotFound(Exception):
    """Custom exception for unfound data"""
    pass

class InputDataField(object):
    """Holds a single datafield as part of the InputData object"""

    def __init__(self, infile, field_name=None):
        """Set filename, check valid, and read the data"""
        self.infile = infile
        self.field_name = field_name

        if None in (infile, field_name):
            raise DataNotFound

        filetype = infile.suffix
        if filetype == '.h5':
            self.read_from_h5()
        else:
            raise NotImplementedError

    def read_from_h5(self):
        """Load data field from HDF5 file"""
        indata = h5py.File(self.infile, 'r')
        try:
            self.xx = indata['x'][:]
            self.yy = indata['y'][:]
            self.field = indata[self.field_name][:]
        except:
            raise DataNotFound


class InputData(object):
    """Loads gridded data & defines interpolators"""

    def __init__(self, params):

        self.params = params
        self.input_dir = params.io.input_dir

        # List of fields to search for
        field_list = ["thick", "bed", "data_mask", "bmelt", "smb", "Bglen"]

        # Dictionary of filenames & field names (i.e. field to get from HDF5 file)
        # Possibly equal to None for variables which have sensible defaults
        # e.g. Basal Melting = 0.0
        self.field_file_dict = {}

        # Dictionary of InputDataField objects for each field
        self.fields = {}

        for f in field_list:
            self.field_file_dict[f] = self.get_field_file(f)
            try:
                self.fields[f] = InputDataField(*self.field_file_dict[f])
            except DataNotFound:
                logging.warning(f"No data found for {f}, "
                                f"field will be filled with default value if appropriate")

        # self.read_data()

    def get_field_file(self, field_name):
        """
        Get the filename & fieldname for a data field from params

        For a given field (e.g. 'thick'), if the parameter
        "thick_data_file" is specified, returns this, otherwise
        assume that the data are in the generic "data_file"
        """
        field_file_str = field_name.lower() + "_data_file"
        field_name_str = field_name.lower() + "_field_name"

        field_filename = self.params.io.__getattribute__(field_file_str)
        field_name = self.params.io.__getattribute__(field_name_str)

        if field_filename is None:
            field_filename = self.params.io.data_file

        if field_filename is None:
            return None, None

        else:
            field_file = Path(self.input_dir)/field_filename
            assert field_file.exists(), f"No input file found for field {field_name}"
            return field_file, field_name

    def interpolate(self, name, space, default=None, static=False):
        """
        Interpolate named variable onto function space

        Arguments:
        name : the variable to be interpolated (need not necessarily exist!)
        space : function space onto which to interpolate
        default : value to return if field is absent (otherwise raise error)
        static : if True, set _Function_static__ = True to save always-zero differentials

        Returns:
        function : the interpolated function
        """

        function = Function(space, name=name, static=static, checkpoint=not static)

        try:
            field = self.fields[name]

        except KeyError:
            # Fill with default, if supplied, else raise error
            if default is not None:
                logging.warning(f"No data found for {name},"
                             f"filling with default value {default}")
                function.vector()[:] = default
                function.vector().apply("insert")
                return function
            else:
                print(f"Failed to find data for field {name}")
                raise

        interper = interp.RegularGridInterpolator((field.xx, field.yy), field.field)
        out_coords = space.tabulate_dof_coordinates()

        result = interper(out_coords)
        function.vector()[:] = result
        function.vector().apply("insert")

        return function

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


#TODO - as yet unused - can't get stdout redirection to work
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)

def setup_logging(params):
    """
    Set up logging to file specified in params
    """

    #TODO - Doesn't work yet - can't redirect output from fenics etc
    # run_name = params.io.run_name
    # logfile = run_name + ".log"

    log_level = params.io.log_level

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    fmt = LogFormatter()

    logger = logging.getLogger("fenics_ice")
    logger.setLevel(numeric_level)

    # Clear out any existing handlers
    # Prevents multiple SO handlers when running multiple phases
    # in same python session/program.
    logger.handlers = []

    so = logging.StreamHandler(sys.stdout)
    so.setFormatter(fmt)
    so.setLevel(numeric_level)

    # fo = logging.FileHandler(logfile)
    # fo.setFormatter(fmt)
    # fo.setLevel(numeric_level)

    # logger.addHandler(fo)
    logger.addHandler(so)

    # sys.stdout = LoggerWriter(logger, numeric_level)
    # sys.stderr = LoggerWriter(logger, numeric_level)

    return logger

    #Consider adding %(process)d- when we move to parallel sims (at least for debug messages?)
#    logging.basicConfig(level=numeric_level, format='%(levelname)s:%(message)s')
#    logging.basicConfig(level=numeric_level, format=)


def print_config(params):

    log = logging.getLogger("fenics_ice")
    log.info("==================================")
    log.info("========= Configuration ==========")
    log.info("==================================\n\n")
    log.info(params)
    log.info("\n\n==================================")
    log.info("======= End of Configuration =====")
    log.info("==================================\n\n")

def log_git_info():
    """Get the current branch & commit hash for logging"""
    repo = git.Repo(__file__, search_parent_directories=True)
    try:
        branch = repo.active_branch.name
    except TypeError:
        branch = "DETACHED"
    sha = repo.head.object.hexsha[:7]

    log = logging.getLogger("fenics_ice")
    log.info("=============== Fenics Ice ===============")
    log.info("==   git branch  : %s" % branch)
    log.info("==   commit hash : %s" % sha)
    log.info("==========================================")

def log_preamble(phase, params):
    """Print out git info, model phase and config"""

    log_git_info()

    log = logging.getLogger("fenics_ice")
    phase_str = f"==  RUNNING {phase.upper()} MODEL PHASE =="
    log.info("\n\n==================================")
    log.info(phase_str)
    log.info("==================================\n\n")

    print_config(params)

