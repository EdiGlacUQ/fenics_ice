# For fenics_ice copyright information see ACKNOWLEDGEMENTS in the fenics_ice
# root directory

# This file is part of fenics_ice.
#
# fenics_ice is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# fenics_ice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

"""
Module to handle model input & output
"""

import sys
import time
import csv
from pathlib import Path
import pickle
import logging
import re
import h5py
import netCDF4
import git
from scipy import interpolate as interp
from abc import ABC, abstractmethod

from fenics import *
from tlm_adjoint.fenics import configure_checkpointing
from collections import defaultdict
import numpy as np

# Regex for catching unnamed vars
unnamed_re = re.compile("f_[0-9]+")

class Writer(ABC):
    """Abstract base class for variable writers"""

    # Regex for catching unnamed vars
    unnamed_re = re.compile("f_[0-9]+")
    suffix = None       # the subclass specific file extension

    def __init__(self, fpath, comm=MPI.comm_world):
        assert comm is not None, "Need an MPI communicator"
        self._fpath = Path(fpath)
        self.comm = comm
        assert self._fpath.suffix == self.suffix

        self.stepped = None      # Single timestep file or multiple?
        self.file_handle = None  # handle for the file
        self.wrote_steps = []

    @staticmethod
    def var_named(var):
        """Does the variable have a name or e.g. f_323"""
        return (unnamed_re.match(var.name()) is None)

    @staticmethod
    def named_copy(var, name):
        """
        Creates a copy of the variable for saving and names it

        If name is None, assume the var is already named and copy it.
        If name is not None, overwrite copy's name.
        """
        new_var = var.copy(deepcopy=True)

        if name is not None:
            new_var.rename(name, "")
        else:
            assert Writer.var_named(var)
            new_var.rename(var.name(), "")

        return new_var

    def open(self):
        self.file_handle = File(str(self._fpath))

    def close(self):
        """No method to close a generic 'File' so just dereference"""
        self.file_handle = None

    def is_open(self):
        return self.file_handle is not None

    def check_step(self, step):
        """
        Check the logic/history of writes to this file

        Output file can have timesteps associated with them or not,
        but this must be used consistently.
        """
        if self.stepped is None:
            self.stepped = step is not None
            return
        else:
            if self.stepped and (step is None):
                raise ValueError("Attempting to write unstepped function to "
                                 "timestepping output file")
            if (not self.stepped) and (step is not None):
                raise ValueError("Attempting to write stepped function to "
                                 "unstepping output file")

    @abstractmethod
    def _write(self, variable, name):
        pass

    def write(self, variable, name=None, step=None, finalise=False):
        """
        Write variable to file

        This method handles the preliminaries, but the actual
        writing is deferred to e.g. VTKWriter & XMDFWriter to define
        """

        # Check file
        if not self.is_open():
            self.open()

        self.check_step(step)

        # Get named variable
        outvar = self.named_copy(variable, name)

        self._write(outvar, step)

        if finalise:
            self.close()

        self.comm.barrier()

class VTKWriter(Writer):
    """Variable writer for .vtk"""

    suffix = '.pvd'

    def check_step(self, step):
        """
        Check that this timestep hasn't been written already (VTK specific)

        fenics' VTK implementation only supports single-variable vtk files,
        so it's only legal to write to a timestep once.
        """
        super().check_step(step)
        if step in self.wrote_steps:
            raise ValueError("Trying to write to existing VTKFile timestep!")

        self.wrote_steps.append(step)

    def _write(self, variable, step):

        if step is None:
            self.file_handle << variable
        else:
            self.file_handle << (variable, step)

class XDMFWriter(Writer):
    """Variable writer for .xdmf"""

    suffix = '.xdmf'
    # stepped = True  # XDMF file components always have a timestep associated

    def _write(self, variable, step):
        if step is None:
            self.file_handle.write(variable, 0)
        else:
            self.file_handle.write(variable, step)

    def open(self):
        """Open XDMFFile (w/ comm)"""
        self.file_handle = XDMFFile(self.comm, str(self._fpath))

    def close(self):
        """Close XDMFFile"""
        self.file_handle.close()
        self.file_handle = None

    def write(self, variable, name=None, step=None, finalise=False):
        """
        Write variable to XDMF file

        This overrides Writer's method because need to
        handle the 'mpi_comm' requirement (get it from variable)
        """
        # TODO - this overloaded write function no longer serves a purpose
        # if the communicator doesn't need to be set
        # if self.comm is None:
        #     self.comm = variable.function_space().mesh().mpi_comm()

        super().write(variable, name, step, finalise)

def gen_path(params, name, suffix):
    """Convert e.g. 'alpha' into outdir/runname_alpha.pvd"""

    outdir = Path(params.io.output_dir)
    outfname = Path("_".join((params.io.run_name, name))).with_suffix(suffix)
    return outdir/outfname


def write_qval(Qval, params):
    """
    Produces pickle dump with QOI value through time
    """

    outdir = params.io.output_dir
    filename = params.io.qoi_file

    run_length = params.time.run_length
    n_steps = params.time.total_steps
    ts = np.linspace(0, run_length, n_steps+1)

    with open(Path(outdir)/filename, 'wb') as pickle_file:
        pickle.dump([Qval, ts], pickle_file)

def write_dqval(dQ_ts, cntrl_names, params):
    """
    Produces .pvd & .h5 files with dQoi_dCntrl
    """

    outdir = params.io.output_dir
    h5_filename = params.io.dqoi_h5file

    vtkfile = File(str((Path(outdir)/h5_filename).with_suffix(".pvd")))

    hdf5out = HDF5File(MPI.comm_world, str(Path(outdir)/h5_filename), 'w')
    n = 0.0

    # Loop dQ sample times ('num_sens')
    for step in dQ_ts:

        assert len(step) == len(cntrl_names)

        # Loop (1 or 2) control vars (alpha, beta)
        for cntrl_name, var in zip(cntrl_names, step):
            output = var
            name = "dQd"+cntrl_name
            output.rename(name, name)
            # vtkfile << output
            hdf5out.write(output, name, n)

        n += 1.0

    hdf5out.close()

def write_variable(var, params, name=None):
    """
    Produce xml & vtk output of supplied variable (prefixed with run name)

    Name is taken from variable structure if not provided
    If 'name' is provided, the variable will be renamed accordingly.
    """
    assert isinstance(var, Function)

    var_name = var.name()
    unnamed_var = unnamed_re.match(var_name) is not None

    outvar = var.copy()  # Copy to avoid persistent rename
    if name is not None:
        pass

    elif unnamed_var:
        # Error if variable is unnamed & no name provided
        logging.error("Attempted to write out an unnamed variable %s" % name)
        raise Exception

    else:
        # Use variable's current name if 'name' not supplied
        name = var_name

    outvar.rename(name, "")

    # Prefix the run name
    outfname = Path(params.io.output_dir)/"_".join((params.io.run_name, name))
    vtk_fname = str(outfname.with_suffix(".pvd"))
    xml_fname = str(outfname.with_suffix(".xml"))

    File(vtk_fname) << outvar
    File(xml_fname) << outvar

    logging.info("Writing function %s to file %s" % (name, outfname))

def dict_to_csv(indict, name, params):
    """Write dictionary to CSV file"""
    outfname = gen_path(params, name, '.csv')
    with open(outfname, 'w') as f:
        writer = csv.DictWriter(f, indict.keys())
        writer.writeheader()
        writer.writerow(indict)


def field_from_vel_file(infile, field_name):
    """Return a field from HDF5 file containing velocity"""
    field = infile[field_name]
    # Check that only one dimension greater than 1 exists
    # i.e. valid: [100], [100,1], [100,1,1], invalid: [100,2]
    assert len([i for i in field.shape if i != 1]) == 1, \
        f"Invalid dimension of field {field_name} from file {infile}"

    return np.ravel(field[:])

def read_vel_obs(params, model=None):
    """
    Read velocity observations & uncertainty from HDF5 file

    For now, expects an HDF5 file
    """
    infile = Path(params.io.input_dir) / params.obs.vel_file
    assert infile.exists(), f"Couldn't find velocity observations file: {infile}"

    infile = h5py.File(infile, 'r')

    # Get grid extent for interpolate via griddata in model.py
    list_extend = list(infile.attrs.keys())
    assert len(list_extend) > 0, \
        f"Invalid velocity file, you need to " \
        f"specify grid extend and spacing for file {infile}"

    extend = defaultdict(list)
    for item in list_extend:
        extend[item].append(infile.attrs[item])

    x_obs = field_from_vel_file(infile, 'x')
    y_obs = field_from_vel_file(infile, 'y')
    u_obs = field_from_vel_file(infile, 'u_obs')
    v_obs = field_from_vel_file(infile, 'v_obs')
    u_std = field_from_vel_file(infile, 'u_std')
    v_std = field_from_vel_file(infile, 'v_std')
    mask_vel = field_from_vel_file(infile, 'mask_vel')

    assert x_obs.size == y_obs.size == u_obs.size == v_obs.size
    assert v_obs.size == u_std.size == v_std.size == mask_vel.size

    uv_obs_pts = np.vstack((x_obs, y_obs)).T

    if model is not None:
        model.uv_obs_pts = uv_obs_pts
        model.u_obs = u_obs
        model.v_obs = v_obs
        model.u_std = u_std
        model.v_std = v_std
        model.mask_vel = mask_vel
        model.extend = extend
    else:
        return uv_obs_pts, u_obs, v_obs, u_std, v_std, mask_vel, extend

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
        assert filetype in [".h5", ".nc"], "Only NetCDF and HDF5 input supported"
        self.read_from_file()

    def read_from_file(self):
        """
        Load data field from HDF5 or NetCDF file

        Expects to find data matrix arranged [y,x], but stores as [x,y]
        """
        filetype = self.infile.suffix
        if filetype == '.h5':
            indata = h5py.File(self.infile, 'r')
        else:
            logging.warning("NetCDF input is untested!")
            indata = netCDF4.Dataset(self.infile, 'r')

        try:
            self.xx = indata['x'][:]
            self.yy = indata['y'][:]
            self.field = indata[self.field_name][:]
        except:
            raise DataNotFound

        # Convert from [y,x] (numpy standard [sort of]) to [x,y]
        self.field = self.field.T

        assert self.field.shape == (self.xx.size, self.yy.size), \
            f"Data have wrong shape! {self.infile}"

        # Take care of data which may be provided y-decreasing, or more rarely
        # x-decreasing...
        if not np.all(np.diff(self.xx) > 0):
            logging.warning(f"Field {self.infile} has x-decreasing - flipping...")
            self.field = np.flipud(self.field)
            self.xx = self.xx[::-1]

        if not np.all(np.diff(self.yy) > 0):
            logging.info(f"Field {self.infile} has y-decreasing - flipping...")
            self.field = np.fliplr(self.field)
            self.yy = self.yy[::-1]

        assert np.unique(np.diff(self.xx)).size == 1,\
            f"{self.infile} not specified on regular grid"
        assert np.unique(np.diff(self.yy)).size == 1,\
            f"{self.infile} not specified on regular grid"

class InputData(object):
    """Loads gridded data & defines interpolators"""

    def __init__(self, params):

        self.params = params
        self.input_dir = params.io.input_dir

        # List of fields to search for
        field_list = ["thick", "bed", "bmelt", "smb", "Bglen", "alpha"]

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

    def interpolate(self, name, space, **kwargs):
        """
        Interpolate named variable onto function space

        Arguments:
        name : the variable to be interpolated (need not necessarily exist!)
        space : function space onto which to interpolate
        default : value to return if field is absent (otherwise raise error)
        static : if True, set _Function_static__ = True to save always-zero differentials
        method: "nearest" or "linear"

        Returns:
        function : the interpolated function

        """
        default = kwargs.get("default", None)
        static = kwargs.get("static", False)
        method = kwargs.get("method", 'linear')
        min_val = kwargs.get("min_val", None)
        max_val = kwargs.get("max_val", None)

        assert (method in ["linear", "nearest"]), f"Unrecognised interpolation method: {method}"
        function = Function(space, name=name, static=static)

        try:
            field = self.fields[name]

        except KeyError:
            # Fill with default, if supplied, else raise error
            if default is not None:
                logging.warning(f"No data found for {name},"
                                f" filling with default value {default}")
                function.vector()[:] = default
                function.vector().apply("insert")
                return function
            else:
                print(f"Failed to find data for field {name}")
                raise

        interper = interp.RegularGridInterpolator((field.xx, field.yy),
                                                  field.field,
                                                  method=method)
        out_coords = space.tabulate_dof_coordinates()

        result = interper(out_coords)

        if (min_val is not None) or (max_val is not None):
            result = np.clip(result, min_val, max_val)

        function.vector()[:] = result
        function.vector().apply("insert")

        return function

# Custom formatter
class LogFormatter(logging.Formatter):
    """A custom formatter for fenics_ice logs"""

    critical_fmt = "CRITICAL ERROR: %(msg)s"
    err_fmt  = "ERROR: %(msg)s"
    warning_fmt  = "WARNING: %(msg)s"
    dbg_fmt  = "DBG: %(module)s: Line %(lineno)d: %(msg)s"
    info_fmt = "%(msg)s"

    def __init__(self):
        """Initialize the parent class"""
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')

    def format(self, record):
        """Format incoming error messages according to log level"""
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


# TODO - as yet unused - can't get stdout redirection to work
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)

def setup_logging(params):
    """Set up logging to file specified in params"""
    # TODO - Doesn't work yet - can't redirect output from fenics etc
    # run_name = params.io.run_name
    # logfile = run_name + ".log"

    # Get the FFC logger to shut up
    logging.getLogger('UFL').setLevel(logging.WARNING)
    logging.getLogger('FFC').setLevel(logging.WARNING)
    logging.getLogger("tlm_adjoint.multistage_checkpointing").setLevel(logging.WARNING)

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

#    Consider adding %(process)d- when we move to parallel sims (at least for debug messages?)
#    logging.basicConfig(level=numeric_level, format='%(levelname)s:%(message)s')
#    logging.basicConfig(level=numeric_level, format=)


def print_config(params):
    """Log the configuration as read from TOML file"""
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
    log.info(f"==   {time.ctime()}   ==")
    log.info("==================================\n\n")

    print_config(params)


def configure_tlm_checkpointing(params):
    """Set up tlm_adjoint's checkpointing"""

    cparam = params.checkpointing
    method = cparam.method

    if method == 'multistage':
        n_steps = params.time.total_steps
        config_dict = {"blocks": n_steps,
                       "snaps_on_disk": cparam.snaps_on_disk,
                       "snaps_in_ram": cparam.snaps_in_ram,
                       "format": "pickle"}

    elif method == 'periodic_disk':
        config_dict = {"period": cparam.period,
                       "format": "pickle"}

    elif method == 'memory':
        config_dict = {}
    else:
        raise ValueError(f"Invalid checkpointing method: {method}")

    configure_checkpointing(method, config_dict)

def write_inversion_info(params, conv_info, header="J, F_crit, G_crit_alpha, G_crit_beta"):
    """Write out a list of tuples containing convergence info for inversion"""
    outfname = Path(params.io.output_dir)/"_".join((params.io.run_name,
                                                    "inversion_progress.csv"))

    np.savetxt(outfname, conv_info, delimiter=",", header=header)
