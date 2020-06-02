"""
Classes & methods for parsing fenics_ice configuration files.
"""

import os
import math
import toml
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from IPython import embed
import pprint

class ConfigPrinter(object):
    """
    Parent class to define how configuration should be printed.
    """
    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            if isinstance(val, dict):
                lines += '{}: dict:\n {}'.format(key, pprint.pformat(val)).split('\n')
            else:
                lines += '{}: {}'.format(key, val).split('\n')
        return '\n    '.join(lines)

class ConfigParser(object):
    """
    A class defining a parser for fenics_ice config files.
    """
    # pylint: disable=too-many-instance-attributes

    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            if key == "config_dict":
                continue
            lines += '{}: {}'.format(key, val).split('\n')
        return '\n    '.join(lines)

    def __init__(self, config_file, top_dir=Path(".")):
        self.top_dir = top_dir  # TODO - hook this up for sims (just plots atm)
        self.config_file = Path(config_file)
        self.config_dict = toml.load(self.config_file)
        self.parse()
        self.check_dirs()

    def parse(self):
        """
        Converts the nested dict self.config_dict into a
        (mostly immutable) structure.
        TODO - check immutibility
        """
        self.io = IOCfg(**self.config_dict['io'])
        self.ice_dynamics = IceDynamicsCfg(**self.config_dict['ice_dynamics'])
        self.inversion = InversionCfg(**self.config_dict['inversion'])
        self.constants = ConstantsCfg(**self.config_dict['constants'])
        self.momsolve = MomsolveCfg(**self.config_dict['momsolve'])
        self.time = TimeCfg(**self.config_dict['time'])
        self.mesh = MeshCfg(**self.config_dict['mesh'])
        self.obs = ObsCfg(**self.config_dict['obs'])
        self.error_prop = ErrorPropCfg(**self.config_dict['errorprop'])
        self.eigendec = EigenDecCfg(**self.config_dict['eigendec'])
        #TODO - boundaries

    def check_dirs(self):
        """
        Check input directory exists & create output dir if necessary.
        """
        assert (self.top_dir / self.io.input_dir).exists(), \
            "Unable to find input directory"

        outdir = (self.top_dir / self.io.output_dir)
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class InversionCfg(ConfigPrinter):
    """
    Configuration related to inversion
    """
    active: bool = False

    max_iter: int = 15
    ftol: float = 1e-4
    gtol: float = None
    verbose: bool = True

    alpha_active: bool = False
    beta_active: bool = False
    simultaneous: bool = False
    alt_iter: int = 2

    gamma_alpha: float = 0.0
    delta_alpha: float = 0.0
    gamma_beta: float = 0.0
    delta_beta: float = 0.0

    def construct_inv_options(self):
        """
        See __post_init__
        """
        inv_options = {"maxiter" : self.max_iter,
                       "disp" : self.verbose,
                       "ftol" : self.ftol,
                       "gtol" : self.gtol
        }
        return cleanNullTerms(inv_options)

    def __post_init__(self):
        """
        Converts supplied parameters (gtol, ftol) etc into a dict for passing
        to minimize_scipy (tlm_adjoint).
        """
        object.__setattr__(self,'inv_options', self.construct_inv_options())

@dataclass(frozen=True)
class ObsCfg(ConfigPrinter):
    """
    Configuration related to observations
    """
    pts_len: float = None

@dataclass(frozen=True)
class ErrorPropCfg(ConfigPrinter):
    """
    Configuration related to error propagation
    """
    qoi: str = 'vaf'

@dataclass(frozen=True)
class EigenDecCfg(ConfigPrinter):
    """
    Configuration related to eigendecomposition
    """
    num_eig: int = None
    eig_algo: str = "slepc"
    power_iter: int = 1   #Number of power iterations for random algorithm
    misfit_only: bool = False
    precondition_by: str = "prior"
    eigenvalue_thresh: float = 1e-1

    def __post_init__(self):
        assert self.precondition_by in ["mass", "prior"], \
            "Valid selections for 'precondition_by' are 'mass' or 'prior'"

        assert self.eig_algo in ["slepc", "random"], \
            "Valid selections for 'eig_algo' are 'slepc' or 'random'"

@dataclass(frozen=True)
class ConstantsCfg(ConfigPrinter):
    """
    Configuration of constants
    """
    rhoi: float = 917.0          #Density of ice
    rhow: float = 1030.0         #Density of sea water
    g: float = 9.81              #Gravity

    ty: float = 365*24*60*60.0   #Year in seconds
    glen_n: float = 3.0          #Exponent in Glen's flow law
    A: float = 3.5e-25 * ty      #Rate factor in Glen's flow law
    eps_rp: float = 1e-5         #Regularisation for strain rate (viscosity)
    vel_rp: float = 1e-2         #Regularisation for velocity
    float_eps: float = 1e-6      #Floats closer than float_eps are considered equal

    random_seed: int = None      #Optionally seeds random generator

    def __post_init__(self):
        """
        Seed the random number generator if seed is specified
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

@dataclass(frozen=True)
class MeshCfg(ConfigPrinter):
    """
    Configuration related to mesh
    """
    mesh_filename: str = None
    data_mesh_filename: str = None
    data_mask_filename: str = None
    nx: int = None
    ny: int = None
    length: float = None
    periodic_bc: bool = False

    def __post_init__(self):
        """
        Check sanity of provided options & set conditional defaults
        Use setattr so dataclass can be frozen
        """

        assert (self.nx is None) == (self.ny is None), \
            "Mesh nx, ny: provide both or neither!"

        assert (self.nx is None) != (self.mesh_filename is None), \
            "Mesh: provide either mesh_filename or nx,ny"

        assert self.data_mask_filename, "Please provide data_mask_filename"

        if self.nx is not None:
            assert self.data_mesh_filename is not None, "Please provide data_mesh_filename"

        #Default filenames
        if self.data_mesh_filename is None:
            object.__setattr__(self, 'data_mesh_filename', "data_mesh.xml")
        if self.data_mask_filename is None: #TODO - check logic here
            object.__setattr__(self, 'data_mask_filename', "data_mask.xml")

@dataclass(frozen=True)
class IceDynamicsCfg(ConfigPrinter):
    """
    Configuration related to modelling ice dynamics
    """
    sliding_law: str = "linear"

    def __post_init__(self):
        assert(self.sliding_law in ['linear', 'weertman'])


@dataclass(frozen=True)
class MomsolveCfg(ConfigPrinter):
    """
    Configuration of MomentumSolver with sensible defaults for picard & newton params
    """

    picard_params: dict = field(default_factory=lambda: {
        'nonlinear_solver': 'newton',
        'newton_solver': {'linear_solver': 'umfpack',
                          'maximum_iterations': 200,
                          'absolute_tolerance': 1.0,
                          'relative_tolerance': 0.001,
                          'convergence_criterion': 'incremental',
                          'error_on_nonconvergence': False}})

    newton_params: dict = field(default_factory=lambda: {
        'nonlinear_solver': 'newton',
        'newton_solver': {'linear_solver': 'umfpack',
                          'maximum_iterations': 25,
                          'absolute_tolerance': 1e-07,
                          'relative_tolerance': 1e-08,
                          'convergence_criterion': 'incremental',
                          'error_on_nonconvergence': True}})


@dataclass(frozen=True)
class IOCfg(ConfigPrinter):
    """
    Configuration parameters for input/output
    """
    run_name: str
    input_dir: str
    output_dir: str

    #TODO - should these be here, or in ErrorPropCfg?
    qoi_file: str = "Qval_ts.p"
    dqoi_h5file: str = "dQ_ts.h5"
    dqoi_vtkfile: str = "dQ_ts.pvd"
    eigenvalue_file: str = "eigvals.p"

    sigma_file: str = "sigma.p"
    sigma_prior_file: str = "sigma_prior.p"

    log_level: str = "info"

    def __post_init__(self):
        assert self.log_level.lower() in ["critical","error","warning","info","debug"], \
            "Invalid log level"

@dataclass(frozen=True)
class TimeCfg(ConfigPrinter):
    """
    Configuration of forward timestepping
    """
    run_length: float
    steps_per_year: float = None #NB: FLOAT
    total_steps: int = None
    dt: float = None
    num_sens: int = 1

    def __post_init__(self):
        """
        Sanity check time configuration
        Logic:
          Must provide run_length (total len)
          Must provide exactly one of: total_steps, dt, steps_per_year
        """
        #Check user provided exactly one way to specify dt:
        assert sum([x is not None for x in [self.total_steps,
                                        self.dt,
                                        self.steps_per_year]]) == 1, \
                                        "Provide one of: dt, total_steps, steps_per_year"

        #Compute other two time measures
        if self.total_steps is not None:
            object.__setattr__(self, 'dt', self.run_length/self.total_steps)
            object.__setattr__(self, 'steps_per_year', 1.0/self.dt)
        elif self.steps_per_year is not None:
            object.__setattr__(self, 'dt', 1.0/self.steps_per_year)
            object.__setattr__(self, 'total_steps', math.ceil(self.run_length/self.dt))
        else: #dt provided
            object.__setattr__(self, 'total_steps', math.ceil(self.run_length/self.dt))
            object.__setattr__(self, 'steps_per_year', 1.0/self.dt)


def cleanNullTerms(d):
    """
    Strips dictionary items where value is None.
    Useful when specifying options to 3rd party lib
    where default should be 'missing' rather than None
    """
    return {
        k:v
        for k, v in d.items()
        if v is not None
    }

#TODO - these are currently unused
newton_defaults_linear = {'nonlinear_solver': 'newton',
                          'newton_solver': {'linear_solver': 'umfpack',
                                            'maximum_iterations': 25,
                                            'absolute_tolerance': 1e-07,
                                            'relative_tolerance': 1e-08,
                                            'convergence_criterion': 'incremental',
                                            'error_on_nonconvergence': True}}

picard_defaults_linear = {'nonlinear_solver': 'newton',
                          'newton_solver': {'linear_solver': 'umfpack',
                                            'maximum_iterations': 200,
                                            'absolute_tolerance': 1.0,
                                            'relative_tolerance': 0.001,
                                            'convergence_criterion': 'incremental',
                                            'error_on_nonconvergence': False}}


newton_defaults_weertman =   {"nonlinear_solver":"newton",
                              "newton_solver":{"linear_solver":"umfpack",
                                               "maximum_iterations":25,
                                               "absolute_tolerance":1.0e-4,
                                               "relative_tolerance":1.0e-5,
                                               "convergence_criterion":"incremental",
                                               "error_on_nonconvergence":True,
                                               "lu_solver":{"same_nonzero_pattern":False,
                                                            "symmetric":False,
                                                            "reuse_factorization":False}}}

picard_defaults_weertman = {"nonlinear_solver":"newton",
                            "newton_solver":{"linear_solver":"umfpack",
                                             "maximum_iterations":200,
                                             "absolute_tolerance":1.0e-4,
                                             "relative_tolerance":1.0e-10,
                                             "convergence_criterion":"incremental",
                                             "error_on_nonconvergence":False,
                                             "lu_solver":{"same_nonzero_pattern":False,
                                                          "symmetric":False,
                                                          "reuse_factorization":False}}}



# infile = "./scripts/run_eg.toml"
# configgy = ConfigParser(infile)
# configgy.parse()
# embed()


