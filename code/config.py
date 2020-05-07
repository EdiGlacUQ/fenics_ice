"""
Classes & methods for parsing fenics_ice configuration files.
"""

import os
import toml
from dataclasses import dataclass, field
from IPython import embed

class ConfigParser(object):
    """
    A class defining a parser for fenics_ice config files.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, config_file):
        self.config_file = config_file
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

    def check_dirs(self):
        """
        Check input directory exists & create output dir if necessary.
        """
        assert os.path.isdir(self.io.input_dir), "Unable to find input directory"
        if not os.path.isdir(self.io.output_dir):
            os.mkdir(self.io.output_dir)

@dataclass(frozen=True)
class InversionCfg(object):
    """
    Configuration related to inversion
    """
    active: bool = False

    max_iter: int = 200
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
class ObsCfg(object):
    """
    Configuration related to inversion
    """
    pts_len: float = None

@dataclass(frozen=True)
class ConstantsCfg(object):
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
class MeshCfg(object):
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

        #TODO - ought we to check that mesh_filename & data_mesh_filename
        #are not both provided?
        assert self.data_mask_filename, "Please provide data_mask_filename"

        if self.nx is not None:
            assert self.data_mesh_filename is not None, "Please provide data_mesh_filename"

        #Set data_mesh_filename = mesh_filename by default
        if self.mesh_filename is not None:
            object.__setattr__(self, 'data_mesh_filename', self.mesh_filename)
        else:
            if self.data_mesh_filename is None:
                object.__setattr__(self, 'data_mesh_filename', "data_mesh.xml")
            if self.data_mask_filename is None: #TODO - check logic here
                object.__setattr__(self, 'data_mask_filename', "data_mask.xml")

@dataclass(frozen=True)
class IceDynamicsCfg(object):
    """
    Configuration related to modelling ice dynamics
    """
    sliding_law: str = "linear"

    def __post_init__(self):
        assert(self.sliding_law in ['linear', 'weertman'])


@dataclass(frozen=True)
class MomsolveCfg(object):
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
class IOCfg(object):
    """
    Configuration parameters for input/output
    """
    run_name: str
    input_dir: str
    output_dir: str

@dataclass(frozen=True)
class TimeCfg(object):
    """
    Configuration of forward timestepping
    """
    run_time: float
    steps_per_year: int = None
    total_steps: int = None
    dt: float = None

    def __post_init__(self):
        assert (bool(self.steps_per_year) != bool(self.total_steps)), \
            "Provide either 'steps_per_year' OR 'total_steps"
        #TODO - could use one to compute the other here...
        assert (bool(self.dt) == bool(self.total_steps))


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


