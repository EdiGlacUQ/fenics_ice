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
Classes & methods for parsing fenics_ice configuration files.
"""

import os
import math
import toml
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import pprint
from fenics import parameters as fenics_params

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
        """Pretty print each member of the ConfigParser object"""
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
        self.set_tlm_adjoint_params()

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

        try:  # Optional BC list
            self.bcs = [BCCfg(**bc) for bc in self.config_dict['BC']]
        except KeyError:
            self.bcs = []
            pass

        try:  # Optional testing
            self.testing = TestCfg(**self.config_dict['testing'])
        except KeyError:
            pass

    def check_dirs(self):
        """
        Check input directory exists & create output dir if necessary.
        """
        assert (self.top_dir / self.io.input_dir).exists(), \
            "Unable to find input directory"

        outdir = (self.top_dir / self.io.output_dir)
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)

    def set_tlm_adjoint_params(self):
        """Set some parameters for tlm_adjoint"""

        # These ensure Jacobian is assembled with the same quadrature rule as
        # the residual, and are required for Newton's method to be second order.
        try:
            fenics_params["tlm_adjoint"]["AssembleSolver"]["match_quadrature"] = True
            fenics_params["tlm_adjoint"]["EquationSolver"]["match_quadrature"] = True
        except RuntimeError:
            print("Warning: unable to set tlm_adjoint param 'match_quadrature'")

@dataclass(frozen=True)
class InversionCfg(ConfigPrinter):
    """
    Configuration related to inversion
    """

    max_iter: int = 15
    ftol: float = None
    gtol: float = None
    verbose: bool = True

    alpha_active: bool = False
    beta_active: bool = False
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

        inv_options = cleanNullTerms(inv_options)
        assert ("ftol" in inv_options) or ("gtol" in inv_options), \
            "Specify either 'ftol' or 'gtol' in inversion options"
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
    vel_file: str = None
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
    mesh_filename: str = 'mesh.xml'
    periodic_bc: bool = False
    bc_filename: str = None

    def __post_init__(self):
        """
        Check sanity of provided options & set conditional defaults
        Use setattr so dataclass can be frozen
        """
        assert Path(self.mesh_filename).suffix in [".xml", ".xdmf"]
        pass

@dataclass(frozen=True)
class BCCfg(ConfigPrinter):
    """Configuration of boundary conditions"""

    labels: tuple  # though it begins as a list...
    flow_bc: str
    name: str = None

    def __post_init__(self):
        """Validate the BC config"""

        possible_types = ["calving", "obs_vel", "no_slip", "free_slip", "natural"]
        assert self.flow_bc in possible_types, f"Unrecognised BC type '{self.flow_bc}'"

        # Convert label list to tuple for immutability
        assert isinstance(self.labels, list)
        assert 0 not in self.labels, "Boundary labels must be positive integers"
        object.__setattr__(self, 'labels', tuple(self.labels))


@dataclass(frozen=True)
class IceDynamicsCfg(ConfigPrinter):
    """Configuration related to modelling ice dynamics"""

    sliding_law: str = "linear"
    min_thickness: float = None

    def __post_init__(self):
        """Check options valid"""
        assert self.sliding_law in ['linear', 'weertman']
        if self.min_thickness is not None:
            assert self.min_thickness >= 0.0


@dataclass(frozen=True)
class MomsolveCfg(ConfigPrinter):
    """
    Configuration of MomentumSolver with sensible defaults for picard & newton params
    """

    picard_params: dict = field(default_factory=lambda: {
        'nonlinear_solver': 'newton',
        'newton_solver': {'linear_solver': 'cg',
                          'preconditioner': 'hypre_amg',
                          'maximum_iterations': 200,
                          'absolute_tolerance': 1.0,
                          'relative_tolerance': 0.001,
                          'convergence_criterion': 'incremental',
                          'error_on_nonconvergence': False}})

    newton_params: dict = field(default_factory=lambda: {
        'nonlinear_solver': 'newton',
        'newton_solver': {'linear_solver': 'cg',
                          'preconditioner': 'hypre_amg',
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

    data_file: str = None

    thick_data_file: str = None
    bed_data_file: str = None
    data_mask_data_file: str = None
    bmelt_data_file: str = None
    smb_data_file: str = None
    bglen_data_file: str = None
    alpha_data_file: str = None

    thick_field_name: str = "thick"
    bed_field_name: str = "bed"
    data_mask_field_name: str = "data_mask"
    bmelt_field_name: str = "bmelt"
    smb_field_name: str = "smb"
    bglen_field_name: str = "Bglen"
    alpha_field_name: str = "alpha"

    inversion_file: str = None
    qoi_file: str = None  # "Qval_ts.p"
    dqoi_h5file: str = None  # "dQ_ts.h5"
    eigenvalue_file: str = None  # "eigvals.p"
    eigenvecs_file: str = None
    sigma_file: str = None  # "sigma.p"
    sigma_prior_file: str = None  # "sigma_prior.p"

    log_level: str = "info"

    def set_default_filename(self, attr_name, suffix):
        """Sets a default filename (prefixed with run_name) & check suffix"""

        # Set default if not set
        fname = self.__getattribute__(attr_name)
        if fname is None:
            object.__setattr__(self,
                               attr_name,
                               '_'.join((self.run_name, suffix)))

        # Check suffix is correct (i.e. if manually set)
        fname = self.__getattribute__(attr_name)
        assert(Path(fname).suffix == Path(suffix).suffix)

    def __post_init__(self):
        """Sanity check & set defaults"""
        assert self.log_level.lower() in ["critical",
                                          "error",
                                          "warning",
                                          "info",
                                          "debug"], \
            "Invalid log level"

        fname_default_suff = {
            'inversion_file': 'invout.h5',
            'eigenvecs_file': 'vr.h5',
            'eigenvalue_file': 'eigvals.p',
            'sigma_file': 'sigma.p',
            'sigma_prior_file': 'sigma_prior.p',
            'qoi_file': 'Qval_ts.p',
            'dqoi_h5file': 'dQ_ts.h5'
        }

        for fname in fname_default_suff:
            self.set_default_filename(fname, fname_default_suff[fname])

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

@dataclass(frozen=True)
class TestCfg(ConfigPrinter):
    """
    Expected values for testing
    """

    expected_J_inv: float = None
    expected_init_alpha: float = None
    expected_cntrl_norm: float = None
    expected_delta_qoi: float = None
    expected_u_norm: float = None
    expected_evals_sum: float = None
    expected_evec0_norm: float = None
    expected_cntrl_sigma_norm: float = None
    expected_cntrl_sigma_prior_norm: float = None
    expected_Q_sigma: float = None
    expected_Q_sigma_prior: float = None


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
