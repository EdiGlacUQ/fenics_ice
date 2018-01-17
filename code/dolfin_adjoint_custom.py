#!/usr/bin/env python3
# -*- coding: utf-8 -*-
  
from fenics import GenericVector as Vector, TestFunction, TrialFunction, \
  action, adjoint, as_backend_type, derivative, dx, has_lu_solver_method, \
  info, norm, parameters, replace
import fenics
import ufl

_dolfin_adjoint = True
if _dolfin_adjoint:
  from dolfin_adjoint import KrylovSolver, LUSolver, assemble, \
    assemble_system, solve
  from dolfin_adjoint import adjglobals, adjlinalg, solving
  import libadjoint
else:
  from fenics import KrylovSolver, LUSolver, assemble, assemble_system, solve
  from adjoint_custom import default_manager

from petsc4py import PETSc

from collections import OrderedDict
import copy
import unittest

"""
This library provides an 'escape hatch' interface for the dolfin-adjoint
library. The construction is principally based on dolfin-adjoint (with code
first added to the repository on 2016-06-02), particularly the dolfin-adjoint
files
  dolfin_adjoint/adjrhs.py
  dolfin_adjoint/assignment.py
  dolfin_adjoint/solving.py
"""

__all__ = \
  [
    "_dolfin_adjoint",
    "AssembleSolver",
    "AssignmentSolver",
    "AxpySolver",
    "Constant",
    "CustomEquation",
    "DACException",
    "DirichletBC",
    "EquationSolver",
    "Function",
    "LinearSolveSolver",
    "NullSolver",
    "TangentLinearMap"
  ]

class DACException(Exception):
  pass
  
class Constant(fenics.Constant):
  def __init__(self, *args, **kwargs):
    kwargs = copy.copy(kwargs)
    static = kwargs.pop("static", False)
    
    fenics.Constant.__init__(self, *args, **kwargs)
    self.__static = static
  
  def is_static(self):
    return self.__static
  
class Function(fenics.Function):
  def __init__(self, *args, **kwargs):
    kwargs = copy.copy(kwargs)
    static = kwargs.pop("static", False)
    
    fenics.Function.__init__(self, *args, **kwargs)
    self.__static = static
  
  def is_static(self):
    return self.__static

class DirichletBC(fenics.DirichletBC):
  def __init__(self, *args, **kwargs):
    kwargs = copy.copy(kwargs)
    static = kwargs.pop("static", False)
    
    fenics.DirichletBC.__init__(self, *args, **kwargs)
    self.__static = static
  
  def is_static(self):
    return self.__static

def is_static_form(form):
  for c in form.coefficients():
    if not hasattr(c, "is_static") or not c.is_static():
      return False
  return True

def is_static_bcs(bcs):
  for bc in bcs:
    if not hasattr(bc, "is_static") or not bc.is_static():
      return False
  return True
  
def next_block_id():
  id = getattr(next_block_id, "id", 0)
  next_block_id.id = id + 1
  return id
  
def assign_vector(x, y):
  x_v = as_backend_type(x).vec()
  if not isinstance(x_v, PETSc.Vec):
    x.zero()
    x.axpy(1.0, y)
    return
  y_v = as_backend_type(y).vec()
  if not isinstance(y_v, PETSc.Vec):
    x.zero()
    x.axpy(1.0, y)
    return
  x_v.setArray(y_v)
  return
  
if _dolfin_adjoint:
  class CustomBlockMatrix(libadjoint.Matrix):
    def __init__(self, A, hermitian):
      self.__A = A
      self.__hermitian = hermitian
      self.__scale = 1.0
      
      return
    
    def function_space(self):
      return self.__A.function_space()
    
    def axpy(self, alpha, x):
      assert(isinstance(self.__A, CustomEquationMatrix))
      assert(isinstance(x, JacobianMatrix))
      self.__A = x
      self.__scale = 1.0 / alpha
    
      return
    
    def solve(self, var, b):
      stop_annotating = parameters["adjoint"]["stop_annotating"]
      parameters["adjoint"]["stop_annotating"] = True

      F = Function(self.function_space())
      F.vector().axpy(self.__scale, b.data.vector())
      b = F.vector()
      
      x = Function(self.function_space())
      self.__A._solve(x, b, self.__hermitian)
      v = adjlinalg.Vector(x)
      
      parameters["adjoint"]["stop_annotating"] = stop_annotating
      return v        

  class CustomBlock(libadjoint.Block):
    def __init__(self, A):
      name = "DAC_%s_%i" % (A.__class__.__name__, next_block_id())
      assert(len(name) <= int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"]) - 1)
      libadjoint.Block.__init__(self, name)
      self.__A = A
      return
    
    def function_space(self):
      return self.__A.function_space()
    
    def assemble(self, dependencies, values, hermitian, coefficient, context):
      assert(coefficient == 1.0)
      return CustomBlockMatrix(self.__A, hermitian), adjlinalg.Vector(Function(self.function_space()))

  class CustomMatrix(libadjoint.Matrix):
    def __init__(self, space):  
      self.__space = space
      
      return
      
    def function_space(self):    
      return self.__space
      
    def block(self):    
      return CustomBlock(self)
      
    def _solve(self, x, b, hermitian):
      raise DACException("Method not overridden")
    
  class IdentityMatrix(CustomMatrix):
    def __init__(self, space):
      CustomMatrix.__init__(self, space)
      return
    
    def _solve(self, x, b, hermitian):    
      assign_vector(x.vector(), b)
      return

  class CustomEquationMatrix(IdentityMatrix):
    def __init__(self, space):
      CustomMatrix.__init__(self, space)
      return
    
  class JacobianMatrix(CustomMatrix):
    def __init__(self, rhs, values, hermitian):
      CustomMatrix.__init__(self, rhs.function_space())
      self.__rhs = rhs
      self.__values = copy.copy(values)
      self.__hermitian = hermitian
      
      return
    
    def _solve(self, x, b, hermitian):
      if not isinstance(x, fenics.Function):
        raise DACException("Unexpected x type: %s" % x.__class__)
      if not isinstance(b, Vector):
        raise DACException("Unexpected b type: %s" % b.__class__)
         
      stop_annotating = parameters["adjoint"]["stop_annotating"]
      parameters["adjoint"]["stop_annotating"] = True
      
      self.__rhs.jacobian_solve(x, self.__values, b, hermitian)
      
      parameters["adjoint"]["stop_annotating"] = stop_annotating
      return

  class CustomRHS(libadjoint.RHS):
    def __init__(self, eq, deps):
      self.__eq = eq
      self.__deps = tuple(deps)
      return
      
    def __call__(self, dependencies, values):
      stop_annotating = parameters["adjoint"]["stop_annotating"]
      parameters["adjoint"]["stop_annotating"] = True
    
      deps = list(self.__eq.dependencies())
      rhs_deps = self.dependencies()
      for dep_var, value in zip(dependencies, values):
        deps[rhs_deps.index(dep_var)] = value.data  
      x = Function(self.function_space())
      self.__eq.forward_solve(x, deps)
      v = adjlinalg.Vector(x)
      
      parameters["adjoint"]["stop_annotating"] = stop_annotating
      return v   
      
    def function_space(self):
      return self.__eq.x().function_space()
      
    def dependencies(self):
      return self.__deps 
      
    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):  
      if not isinstance(contraction_vector.data, fenics.Function):
        raise DACException("Unexpected contraction_vector.data type: %s" % contraction_vector.data.__class__)
      
      stop_annotating = parameters["adjoint"]["stop_annotating"]
      parameters["adjoint"]["stop_annotating"] = True
      
      b = self.__eq.derivative_action([value.data for value in values],
        dependencies.index(variable),
        contraction_vector.data, hermitian)
      if isinstance(b, Vector):
        if hermitian:
          F = Function(values[dependencies.index(variable)].data.function_space())
        else:
          F = Function(self.function_space())
        assign_vector(F.vector(), b)
        b = F
      v = adjlinalg.Vector(b)
      
      parameters["adjoint"]["stop_annotating"] = stop_annotating
      return v
    
    def second_derivative_action(self, dependencies, values, inner_variable, inner_contraction_vector, outer_variable, hermitian, action_vector):
      if not isinstance(inner_contraction_vector.data, fenics.Function):
        raise DACException("Unexpected inner_contraction_vector.data type: %s" % inner_contraction_vector.data.__class__)
      if not isinstance(action_vector.data, fenics.Function):
        raise DACException("Unexpected action_vector.data type: %s" % action_vector.data.__class__)
        
      stop_annotating = parameters["adjoint"]["stop_annotating"]
      parameters["adjoint"]["stop_annotating"] = True
   
      inner_index = dependencies.index(inner_variable)
      outer_index = dependencies.index(outer_variable)   
      b = self.__eq.second_derivative_action([value.data for value in values],
        dependencies.index(inner_variable),
        inner_contraction_vector.data,
        dependencies.index(outer_variable),
        action_vector.data,
        hermitian)
      if isinstance(b, Vector):
        F = Function(values[outer_index if hermitian else inner_index].data.function_space())
        assign_vector(F.vector(), b)
        b = F
      v = adjlinalg.Vector(b)
      
      parameters["adjoint"]["stop_annotating"] = stop_annotating
      return v
    
    def derivative_assembly(self, dependencies, values, variable, hermitian):
      return JacobianMatrix(self, [value.data for value in values], hermitian)
    
    def jacobian_solve(self, *args, **kwargs):
      return self.__eq.jacobian_solve(*args, **kwargs)

class CustomEquation:
  def __init__(self, x, deps):
    """
    A custom equation. This provides an 'escape hatch' via which arbitrary
    equations can be solved and recorded appropriately on the libadjoint tape.
    The equation is expressed in the form:
      F ( x, y_0, y_1, ... ) = 0,
    where x is the equation solution and F is a RHS. Information regarding F is
    provided by the methods apply_bcs, apply_hbcs, set_bc_dofs, residual,
    forward_solve, derivative_action, and jacobian_solve, which should be
    overridden as required by derived classes.
    
    Arguments:
      x     A Function. The solution to the equation.
      deps  A list of Function dependencies, which must include x.
    """

    if not x in deps:
      raise DACException("Equation must have the solution as a dependency")
    if not len(set(deps)) == len(deps):
      raise DACException("Duplicate dependency")

    if _dolfin_adjoint: 
      self.__A_block = CustomEquationMatrix(x.function_space()).block()
    self.__x = x
    self.__deps = tuple(deps)
    
    return
    
  def replace(self):
    """
    Replace all internal Function objects.
    """
  
    if not _dolfin_adjoint:
      default_manager.replace(self)
  
  def _replace(self, replace_map):
    """
    Replace all internal Function objects using the supplied replace map. Must
    call the base class _replace method.
    """
    
    self.__x = replace_map.get(self.__x, self.__x)
    self.__deps = tuple(replace_map.get(dep, dep) for dep in self.__deps)
    
    return
    
  def x(self):
    """
    A Function. The solution to the equation.
    """
  
    return self.__x

  def function_space(self):
    """
    The solution function space.
    """

    return self.x().function_space()
  
  def dependencies(self):
    return self.__deps
    
  def _pre_annotate(self):
    if _dolfin_adjoint:
      # Based on 'annotate' in dolfin-adjoint file dolfin_adjoint/solving.py
      # (see dolfin-adjoint version 2017.1.0)
      # Code first added 2017-12-18
      
      x = self.x()
      var = adjglobals.adj_variables[x]
      solving.register_initial_conditions([[x, var]], linear = False)  
    else:
      default_manager.add_initial_condition(self.x())
      
    return
      
  def _post_annotate(self, replace = False):    
    if _dolfin_adjoint:
      x = self.x()
      deps = self.dependencies()
      
      # Based on 'annotate' in dolfin-adjoint file dolfin_adjoint/solving.py
      # Code first added to repository 2016-06-02

      var = adjglobals.adj_variables[x]
      # 2017-12-18: Now handled by _pre_annotate above
      #solving.register_initial_conditions([[x, var]], linear = False)  
      var = adjglobals.adj_variables.next(x)

      dep_vars = [adjglobals.adj_variables[dep] for dep in deps]
      solving.register_initial_conditions(zip(deps, dep_vars), linear = False, var = var)   
             
      rhs = CustomRHS(self, dep_vars)
      eq = libadjoint.Equation(var, blocks = [self.__A_block], targets = [var], rhs = rhs)
        
      cs = adjglobals.adjointer.register_equation(eq)
      solving.do_checkpoint(cs, var, rhs)
      
      # As in 'register_assign' in dolfin-adjoint file
      # dolfin_adjoint/assignment.py
      # Code first added to repository 2016-06-02
      if parameters["adjoint"]["record_all"]:
        adjglobals.adjointer.record_variable(var, libadjoint.MemoryStorage(adjlinalg.Vector(x)))
    else:
      default_manager.annotate(self, replace = replace)

    return
    
  def solve(self, annotate = None, replace = False):
    """
    Solve the equation.
    
    Arguments:
    
    annotate  (Optional) Whether the equation should be recorded on the
              libadjoint tape. Overrides the dolfin-adjoint 'stop_annotating'
              parameter.
    replace   (Optional) Whether, after the equation has been solved, its
              internal Function objects should be replaced with Coefficients.
              Can be used to save memory in the annotation.
    """
    
    if annotate is None:
      annotate = not parameters["adjoint"]["stop_annotating"]
  
    stop_annotating = parameters["adjoint"]["stop_annotating"]
    parameters["adjoint"]["stop_annotating"] = True
    
    if annotate:
      self._pre_annotate()
    self.forward_solve(self.x(), self.dependencies())
    if annotate:
      self._post_annotate(replace = replace)
      
    parameters["adjoint"]["stop_annotating"] = stop_annotating
    return
  
  def norm(self, x):
    """
    Return the norm of a Function in the function space associated with the
    equation. Defaults to the L^2 norm, but can be overridden by derived
    classes.
    
    Arguments:
      x  A Function. Function for which the norm should be computed.
    """
    
    return norm(x, "l2")
    
  def dual_norm(self, b):
    """
    Return the norm of an assembled rank one form in the dual space associated
    with the equation function space. Defaults to the simple Euclidean 2-norm
    of the coefficients, but can be overridden by derived classes.
    
    Arguments:
      b  A Vector. Assembled rank one form.
    """
  
    return b.norm("l2")
    
  def apply_bcs(self, v):
    """
    Apply Dirichlet boundary conditions associated with this equation to a given
    Vector.
    """
  
    raise DACException("Method not overridden")
    
  def apply_hbcs(self, v):
    """
    Apply homogenised Dirichlet boundary conditions associated with this
    equation to a given Vector.
    """
  
    raise DACException("Method not overridden")

  def forward_solve(self, x, deps):
    """
    Solve the equation. The libadjoint tape is disabled when this method is
    called.
    
    Arguments:
      x     A Function. The solution, which should be set by this method.
      deps  A list of Function objects defining the values of dependencies.
    """
  
    raise DACException("Method not overridden")
    
  def reset_forward_solve(self):
    """
    Reset the forward solver. Can be used to clear caches used by forward_solve.
    """
    
    return
    
  def derivative_action(self, deps, index, u, hermitian):
    """
    Return a Function, Vector, or Form containing the action of the derivative
    of the RHS, or None if this is zero. The libadjoint tape is disabled when
    this method is called.
    
    Arguments:
      deps       A list of Function objects defining the values of dependencies.
      index      The index of the dependency in deps with respect to which
                 a derivative should be taken.
      u          A Function defining the direction in which the derivative
                 should be evaluated.
      hermitian  If True then the adjoint derivative action is returned.
    """
  
    raise DACException("Method not overridden")
  
  def second_derivative_action(self, deps, inner_index, inner_u, outer_index, u, hermitian):
    """
    Return a Function, Vector, or Form containing the action of the second
    derivative of the RHS, or None if this is zero. This libadjoint tape is
    dsiabled when this method is called.
    
    Arguments:
      deps         A list of Function objects defining the values of
                   dependencies.
      inner_index  The index of the dependency in deps with respect to which
                   the first derivative should be taken.
      inner_u      A Function defining the direction in which the first
                   derivative should be taken.
      outer_index  The index of the dependency in deps with respect to which
                   the second derivative should be taken.
      u            A Function defining the direction in which the second
                   derivative should be evaluated.
      hermitian    If True then the adjoint second derivative action is
                   returned.
    """
  
    raise DACException("Method not overridden")

  def jacobian_solve(self, u, deps, b, hermitian):
    """
    Solve a tangent linear or adjoint equation. The libadjoint tape is disabled
    when this method is called.
    
    Arguments:
      u          A Function. The solution, which should be set by this method.
      deps       A list of Function objects defining the values of dependencies.
      b          A Vector defining the right-hand-side. May be modified.
      hermitian  If True then the equation solved is (dF/dx)^* u = b. Otherwise
                 the equation solved is dF/dx u = b.
    """
  
    raise DACException("Method not overridden")
    
  def reset_jacobian_solve(self):
    """
    Reset the Jacobian solver. Can be used to clear caches used by
    jacobian_solve.
    """
    
    return
  
  def tangent_linear(self, m, tlm_map):
    """
    Return a CustomEquation corresponding to a tangent linear equation.
    
    Arguments:
      m        The parameter defining the tangent-linear equation.
      tlm_map  The TangentLinearMap.
    """
  
    raise DACException("Method not overridden")

class TangentLinearMap:
  """
  A map from forward to TLM variables.
  """

  def __init__(self):
    self.clear()
  
  def __contains__(self, x):
    return x in self.__map
  
  def __getitem__(self, x):
    if not x in self.__map:
      self.__map[x] = Function(x.function_space(), name = "DAC_%s_tlm" % x.name())
    return self.__map[x]
  
  def clear(self):
    self.__map = OrderedDict()

class NullSolver(CustomEquation):
  def __init__(self, x):
    CustomEquation.__init__(self, x, [x])
    return
    
  def _pre_annotate(self):
    return

  def apply_bcs(self, v):
    return
    
  def apply_hbcs(self, v):
    return

  def forward_solve(self, x, deps):
    x.vector().zero()
    return
    
  def derivative_action(self, deps, index, u, hermitian):
    if index == 0:
      return u
    else:
      return None
  
  def second_derivative_action(self, deps, inner_index, inner_u, outer_index, u, hermitian):
    return None
      
  def jacobian_solve(self, u, deps, b, hermitian):
    assign_vector(u.vector(), b)
    return
    
  def tangent_linear(self, m, tlm_map):
    return NullSolver(tlm_map[self.x()])

class AssignmentSolver(CustomEquation):
  def __init__(self, y, x, bcs = []):
    if x == y:
      raise DACException("Non-linear dependency in linear equation")
    
    hbcs = [DirichletBC(bc) for bc in bcs]
    [hbc.homogenize() for hbc in hbcs]
    
    CustomEquation.__init__(self, x, [x, y])  # TODO: Consider boundary conditions
    self.__bcs = copy.copy(bcs)
    self.__hbcs = hbcs
    return
    
  def _pre_annotate(self):
    return

  def apply_bcs(self, v):
    [bc.apply(v) for bc in self.__bcs]
    return
    
  def apply_hbcs(self, v):
    [bc.apply(v) for bc in self.__hbcs]
    return

  def forward_solve(self, x, deps):
    assign_vector(x.vector(), deps[1].vector())
    self.apply_bcs(x.vector())
    return
    
  def derivative_action(self, deps, index, u, hermitian):
    if index == 0:
      return u
    elif index == 1:
      F = Function(self.function_space())
      F.vector().axpy(-1.0, u.vector())
      return F
    else:
      return None
  
  def second_derivative_action(self, deps, inner_index, inner_u, outer_index, u, hermitian):
    return None
      
  def jacobian_solve(self, u, deps, b, hermitian):
    assign_vector(u.vector(), b)
    self.apply_hbcs(u.vector())
    return
    
  def tangent_linear(self, m, tlm_map):
    x, y = self.dependencies()
    if m in [x, y]:
      raise DACException("Invalid tangent-linear parameter")    
    elif not y in tlm_map:
      return NullSolver(tlm_map[x])
    else:
      return AssignmentSolver(tlm_map[y], tlm_map[x], bcs = self.__hbcs)

class AxpySolver(CustomEquation):
  def __init__(self, x_old, alpha, y, x_new, bcs = []):
    if x_new in [x_old, y]:
      raise DACException("Non-linear dependency in linear equation")
    
    hbcs = [DirichletBC(bc) for bc in bcs]
    [hbc.homogenize() for hbc in hbcs]
    
    CustomEquation.__init__(self, x_new, [x_new, x_old, y])  # TODO: Consider boundary conditions
    self.__alpha = float(alpha)
    self.__bcs = copy.copy(bcs)
    self.__hbcs = hbcs
    return

  def apply_bcs(self, v):
    [bc.apply(v) for bc in self.__bcs]
    return
    
  def apply_hbcs(self, v):
    [bc.apply(v) for bc in self.__hbcs]
    return

  def forward_solve(self, x, deps):
    x.assign(deps[1])
    x.vector().axpy(self.__alpha, deps[2].vector())
    self.apply_bcs(x.vector())
    return
    
  def derivative_action(self, deps, index, u, hermitian):
    if index == 0:
      return u
    elif index == 1:
      F = Function(self.function_space())
      F.vector().axpy(-1.0, u.vector())
      return F
    elif index == 2:
      F = Function(self.function_space())
      F.vector().axpy(-self.__alpha, u.vector())
      return F
    else:
      return None
  
  def second_derivative_action(self, deps, inner_index, inner_u, outer_index, u, hermitian):
    return None
      
  def jacobian_solve(self, u, deps, b, hermitian):
    assign_vector(u.vector(), b)
    self.apply_hbcs(u.vector())
    return

class AssembleSolver(CustomEquation):
  def __init__(self, rhs, x, bcs = [], form_compiler_parameters = None):
    deps = set()
    for dep in rhs.coefficients():
      if isinstance(dep, fenics.Function):
        deps.add(dep)
    # TODO: Consider other Coefficient types and boundary conditions
    if x in deps:
      raise DACException("Non-linear dependency in linear equation")
    deps = list(deps)
    deps.insert(0, x)
    deps[1:] = sorted(deps[1:], key = lambda dep : dep.id())
    
    hbcs = [DirichletBC(bc) for bc in bcs]
    [hbc.homogenize() for hbc in hbcs]
    
    CustomEquation.__init__(self, x, deps)
    self.__rhs = rhs
    self.__bcs = copy.copy(bcs)
    self.__hbcs = hbcs
    self.__form_compiler_parameters = copy.deepcopy(form_compiler_parameters)

  def _replace(self, replace_map):
    CustomEquation._replace(self, replace_map)
    self.__rhs = replace(self.__rhs, replace_map)

  def apply_bcs(self, v):
    [bc.apply(v) for bc in self.__bcs]
    return

  def apply_hbcs(self, v):
    [bc.apply(v) for bc in self.__hbcs]
    return

  def forward_solve(self, x, deps):
    replace_map = dict(zip(self.dependencies(), deps))
    assemble(replace(self.__rhs, replace_map),
      form_compiler_parameters = self.__form_compiler_parameters,
      tensor = x.vector())
    self.apply_bcs(x.vector())
    return
    
  def derivative_action(self, deps, index, u, hermitian):    
    # Derived from EquationSolver.derivative_action (see dolfin-adjoint
    # reference below). Code first added 2017-12-07.
    
    if index == 0:
      return u
    else:
      try:
        zeta = self.dependencies()[index]
      except IndexError:
        return None
      J = derivative(self.__rhs, zeta, du = TrialFunction(zeta.function_space()))
      J = ufl.algorithms.expand_derivatives(J)
      if len(J.integrals()) == 0:
        return None
      if hermitian:
        J = adjoint(J)
      replace_map = dict(zip(self.dependencies(), deps))
      J = replace(J, replace_map)
          
#       return action(-J, u)
      return assemble(action(-J, u), form_compiler_parameters = self.__form_compiler_parameters)
  
  def second_derivative_action(self, deps, inner_index, inner_u, outer_index, u, hermitian):
    # Derived from EquationSolver.second_derivative_action (see dolfin-adjoint
    # reference below). Code first added 2017-12-07.
      
    try:
      inner_zeta = self.dependencies()[inner_index]
      outer_zeta = self.dependencies()[outer_index]
    except IndexError:
      return None
    H = derivative(self.__rhs, inner_zeta, du = inner_u)
    H = ufl.algorithms.expand_derivatives(H)
    if len(H.integrals()) == 0:
      return None
    H = derivative(H, outer_zeta, du = TrialFunction(outer_zeta.function_space()))
    H = ufl.algorithms.expand_derivatives(H)
    if len(H.integrals()) == 0:
      return None
    if hermitian:
      H = adjoint(H)
    replace_map = dict(zip(self.dependencies(), deps))
    H = replace(H, replace_map)

#     return action(-H, u)
    return assemble(action(-H, u), form_compiler_parameters = self.__form_compiler_parameters)
      
  def jacobian_solve(self, u, deps, b, hermitian):
    assign_vector(u.vector(), b)
    self.apply_hbcs(u.vector())
    return
  
  def tangent_linear(self, m, tlm_map):
    x = self.x()
    if m == x:
      raise DACException("Invalid tangent-linear parameter")
      
    tlm_rhs = derivative(self.__rhs, m, du = Constant(1.0, static = True))
    for dep in self.dependencies():
      if dep != x and dep in tlm_map:
        tlm_rhs += derivative(self.__rhs, dep, du = tlm_map[dep])
    
    return AssembleSolver(tlm_rhs, tlm_map[x], self.__hbcs,
      form_compiler_parameters = self.__form_compiler_parameters)

class LinearSolveSolver(CustomEquation):
  def __init__(self, lhs, x, rhs, bcs = [],
    solver_parameters = {"linear_solver":"default"},
    form_compiler_parameters = {},
    initial_guess = None):
    # TODO: Jacobian caching
   
    deps = set()
    for dep in lhs.coefficients():
      if isinstance(dep, fenics.Function):
        deps.add(dep)
      # TODO: Consider other Coefficient types and boundary conditions
    if not initial_guess is None:
      if initial_guess == x:
        raise DACException("Cannot use solution as an initial guess")
      elif initial_guess == rhs:
        raise DACException("Cannot use rhs as initial guess")
      deps.add(initial_guess)
    if x in deps:
      raise DACException("Non-linear dependency in linear equation")
    elif rhs in deps:
      raise DACException("Rhs dependency in lhs")
    deps = list(deps)
    deps.insert(0, x)
    deps.insert(1, rhs)
    deps[2:] = sorted(deps[2:], key = lambda dep : dep.id())
    
    hbcs = [DirichletBC(bc) for bc in bcs]
    [hbc.homogenize() for hbc in hbcs]
    if len(bcs) > 0:
      dummy_rhs = TestFunction(lhs.arguments()[0].function_space()) * Constant(0.0) * dx
    else:
      dummy_rhs = None
    
    CustomEquation.__init__(self, x, deps)
    self.__lhs, self.__dummy_rhs = lhs, dummy_rhs
    self.__bcs = copy.copy(bcs)
    self.__hbcs = hbcs
    self.__form_compiler_parameters = copy.deepcopy(form_compiler_parameters)
    self.__linear_solver_parameters = copy.deepcopy(solver_parameters)
    self.__initial_guess = initial_guess
    
    return

  def _replace(self, replace_map):
    CustomEquation._replace(self, replace_map)
    self.__lhs = replace(self.__lhs, replace_map)

  def apply_bcs(self, v):
    [bc.apply(v) for bc in self.__bcs]
    return

  def apply_hbcs(self, v):
    [bc.apply(v) for bc in self.__hbcs]
    return

  def forward_solve(self, x, deps):
    replace_map = dict(zip(self.dependencies(), deps))    
    
    if not self.__initial_guess is None:
      x.assign(replace_map[self.__initial_guess])
    replace_map[self.x()] = x
    
    lhs = replace(self.__lhs, replace_map)
    if len(self.__bcs) > 0:
      J, b_bcs = assemble_system(lhs, self.__dummy_rhs, self.__bcs, form_compiler_parameters = self.__form_compiler_parameters)
      b = deps[1].vector().copy()
      self.apply_hbcs(b)
      b.axpy(1.0, b_bcs)
    else:
      J = assemble(lhs, form_compiler_parameters = self.__form_compiler_parameters)
      b = deps[1].vector()
      
    J_solver = _linear_solver(self.__linear_solver_parameters)
    J_solver.set_operator(J)        
    J_solver.solve(x.vector(), b)
    return
    
  def derivative_action(self, deps, index, u, hermitian):
    # Derived from EquationSolver.derivative_action (see dolfin-adjoint
    # reference below). Code first added 2017-12-07.
    
    if index == 1:
      F = Function(self.function_space())
      F.vector().axpy(-1.0, u.vector())
      return F
    else:
      try:
        zeta = self.dependencies()[index]
      except IndexError:
        return None
      J = derivative(action(self.__lhs, self.x()), zeta, du = TrialFunction(zeta.function_space()))
      J = ufl.algorithms.expand_derivatives(J)
      if len(J.integrals()) == 0:
        return None
      if hermitian:
        J = adjoint(J)
      replace_map = dict(zip(self.dependencies(), deps))
      J = replace(J, replace_map)
          
  #     return action(J, u)
      return assemble(action(J, u), form_compiler_parameters = self.__form_compiler_parameters)
  
  def second_derivative_action(self, deps, inner_index, inner_u, outer_index, u, hermitian):
    # Derived from EquationSolver.second_derivative_action (see dolfin-adjoint
    # reference below)
    
    if 1 in [inner_index, outer_index]:
      return None  

    try:
      inner_zeta = self.dependencies()[inner_index]
      outer_zeta = self.dependencies()[outer_index]
    except IndexError:
      return None
    H = derivative(action(self.__rhs, self.x()), inner_zeta, du = inner_u)
    H = ufl.algorithms.expand_derivatives(H)
    if len(H.integrals()) == 0:
      return None
    H = derivative(H, outer_zeta, du = TrialFunction(outer_zeta.function_space()))
    H = ufl.algorithms.expand_derivatives(H)
    if len(H.integrals()) == 0:
      return None
    if hermitian:
      H = adjoint(H)
    replace_map = dict(zip(self.dependencies(), deps))
    H = replace(H, replace_map)

#     return action(H, u)
    return assemble(action(H, u), form_compiler_parameters = self.__form_compiler_parameters)

  def jacobian_solve(self, u, deps, b, hermitian):
    replace_map = dict(zip(self.dependencies(), deps))   
    J = replace(self.__lhs, replace_map)
    if hermitian:
      J = adjoint(J)
    
    if len(self.__bcs) > 0:
      J, _ = assemble_system(J, self.__dummy_rhs, self.__hbcs, form_compiler_parameters = self.__form_compiler_parameters)
      self.apply_hbcs(b)
    else:
      J = assemble(J, form_compiler_parameters = self.__form_compiler_parameters)
    
    J_solver = _linear_solver(self.__linear_solver_parameters)
    J_solver.set_operator(J)
    J_solver.solve(u.vector(), b)
    
    return
    
def _linear_solver(linear_solver_parameters):
  linear_solver = linear_solver_parameters.get("linear_solver", "default")
  if linear_solver in ["direct", "lu"]:
    linear_solver = "default"
  elif linear_solver == "iterative":
    linear_solver = "gmres"
  is_lu_linear_solver = linear_solver == "default" or has_lu_solver_method(linear_solver)
  if is_lu_linear_solver:
    solver = LUSolver(linear_solver)
    solver.parameters.update(linear_solver_parameters.get("lu_solver", {}))
  else:
    solver = KrylovSolver(linear_solver, linear_solver_parameters.get("preconditioner", "none"))
    solver.parameters.update(linear_solver_parameters.get("krylov_solver", {}))
  return solver
   
class EquationSolver(CustomEquation):
  def __init__(self, *args, **kwargs):
    kwargs = copy.copy(kwargs)
    initial_guess = kwargs.pop("initial_guess", None)
    cache_jacobian = kwargs.pop("cache_jacobian", None)  # Default value set below
    eq, x, bcs, J, _, _, form_compiler_parameters, solver_parameters = fenics.fem.solving._extract_args(*args, **kwargs)
    if not J is None:
      raise DACException("Custom Jacobian not supported")
    
    lhs, rhs = eq.lhs, eq.rhs
    deps = set()
    for dep in lhs.coefficients():
      if isinstance(dep, fenics.Function):
        deps.add(dep)
      # TODO: Consider other Coefficient types and boundary conditions
    if not rhs == 0:
      for dep in rhs.coefficients():
        if isinstance(dep, fenics.Function):
          deps.add(dep)
        # TODO: Consider other Coefficient types and boundary conditions
    if not initial_guess is None:
      if initial_guess == x:
        raise DACException("Cannot use solution as an initial guess")
      deps.add(initial_guess)
    deps = list(deps) 
    
    if isinstance(lhs, ufl.form.Form) and isinstance(rhs, ufl.form.Form):
      if x in deps:
        raise DACException("Non-linear dependency in linear equation")
      deps.insert(0, x)
      F = action(lhs, x) - rhs
      J = lhs
      linear = True
    else:
      while x in deps:
        deps.remove(x)
      deps.insert(0, x)
      F = lhs
      if not rhs == 0:
        F -= rhs
      J = derivative(F, x, du = TrialFunction(x.function_space()))
      linear = not x in J.coefficients()
    deps[1:] = sorted(deps[1:], key = lambda dep : dep.id())
    
    hbcs = [DirichletBC(bc) for bc in bcs]
    [hbc.homogenize() for hbc in hbcs]
    
    if cache_jacobian is None and linear and is_static_form(J) and is_static_bcs(bcs):
      cache_jacobian = True
    
    def update_parameters(old_parameters, new_parameters):
      for key, value in new_parameters.items():
        if not key in old_parameters:
          old_parameters[key] = copy.deepcopy(value)
        elif isinstance(old_parameters[key], dict):
          if not isinstance(value, dict):
            raise DACException("Invalid solver parameter: %s" % key)
          update_parameters(old_parameters[key], value)
        else:
          old_parameters[key] = old_parameters[key].__class__(value)
      return
    if cache_jacobian:
      linear_solver_parameters = {"lu_solver":{"reuse_factorization":True, "same_nonzero_pattern":True},
                                  "krylov_solver":{}}
    else:
      linear_solver_parameters = {}
    if linear:
      update_parameters(linear_solver_parameters, solver_parameters)
    else:
      nl_solver = solver_parameters.get("nonlinear_solver", "newton")
      if nl_solver == "newton":
        update_parameters(linear_solver_parameters, solver_parameters.get("newton_solver", {}))
      else:
        raise DACException("Unsupported non-linear solver: %s" % nl_solver)
    
    CustomEquation.__init__(self, x, deps)    
    self.__F = F
    self.__lhs, self.__rhs = lhs, rhs
    self.__bcs = copy.copy(bcs)
    self.__hbcs = hbcs
    self.__J = J
    self.__form_compiler_parameters = copy.deepcopy(form_compiler_parameters)
    self.__solver_parameters = copy.deepcopy(solver_parameters)
    self.__linear_solver_parameters = linear_solver_parameters
    self.__initial_guess = initial_guess
    self.__linear = linear
    
    self.__cache_jacobian = cache_jacobian
    self.reset_forward_solve()
    self.reset_jacobian_solve()
    
    return

  def _replace(self, replace_map):
    CustomEquation._replace(self, replace_map)
    self.__F = replace(self.__F, replace_map)
    self.__lhs = replace(self.__lhs, replace_map)
    if not self.__rhs == 0:
      self.__rhs = replace(self.__rhs, replace_map)
    self.__J = replace(self.__J, replace_map)

  def apply_bcs(self, v):
    [bc.apply(v) for bc in self.__bcs]
    return

  def apply_hbcs(self, v):
    [bc.apply(v) for bc in self.__hbcs]
    return

  def forward_solve(self, x, deps):
    replace_map = dict(zip(self.dependencies(), deps))    
    
    if not self.__initial_guess is None:
      x.assign(replace_map[self.__initial_guess])    
    replace_map[self.x()] = x
    
    lhs = replace(self.__lhs, replace_map)
    rhs = 0 if self.__rhs == 0 else replace(self.__rhs, replace_map)
    J = replace(self.__J, replace_map)
    
    if self.__linear:
      if self.__forward_J_mat is None:
        if self.__cache_jacobian:
          dummy_rhs = TestFunction(lhs.arguments()[0].function_space()) * Constant(0.0) * dx
          J, b_bc = assemble_system(lhs, dummy_rhs, self.__bcs, form_compiler_parameters = self.__form_compiler_parameters)
          b = assemble(rhs, form_compiler_parameters = self.__form_compiler_parameters)
          self.apply_hbcs(b)
          b.axpy(1.0, b_bc)
          self.__forward_J_mat = J, b_bc
        else:
          J, b = assemble_system(lhs, rhs, self.__bcs, form_compiler_parameters = self.__form_compiler_parameters)
      else:
        J = self.__forward_J_mat[0]
        b = assemble(rhs, form_compiler_parameters = self.__form_compiler_parameters)
        self.apply_hbcs(b)
        b.axpy(1.0, self.__forward_J_mat[1])
      
      if self.__forward_J_solver is None:
        J_solver = _linear_solver(self.__linear_solver_parameters)
        J_solver.set_operator(J)
        self.__forward_J_solver = J_solver
      else:
        J_solver = self.__forward_J_solver
        if not self.__cache_jacobian:
          J_solver.set_operator(J)
        
      J_solver.solve(x.vector(), b)
    else:
      solve(lhs == rhs, x, self.__bcs, J = J, form_compiler_parameters = self.__form_compiler_parameters, solver_parameters = self.__solver_parameters)
    
    return
    
  def reset_forward_solve(self):
    self.__forward_J_mat = None
    self.__forward_J_solver = None
    
  def derivative_action(self, deps, index, u, hermitian):
    # Similar to 'RHS.derivative_action' and 'RHS.second_derivative_action' in
    # dolfin-adjoint file dolfin_adjoint/adjrhs.py
    # Code first added to repository 2016-06-02
    
    try:
      zeta = self.dependencies()[index]
    except IndexError:
      return None
    J = derivative(self.__F, zeta, du = TrialFunction(zeta.function_space()))
    J = ufl.algorithms.expand_derivatives(J)
    if len(J.integrals()) == 0:
      return None
    if hermitian:
      J = adjoint(J)
    replace_map = dict(zip(self.dependencies(), deps))
    J = replace(J, replace_map)
        
#     return action(J, u)
    return assemble(action(J, u), form_compiler_parameters = self.__form_compiler_parameters)
  
  def second_derivative_action(self, deps, inner_index, inner_u, outer_index, u, hermitian):
    # As in 'RHS.second_derivative_action' in dolfin-adjoint file
    # dolfin_adjoint/adjrhs.py (see dolfin-adjoint version 2017.1.0)
    # Code first added 2017-12-01
    
    try:
      inner_zeta = self.dependencies()[inner_index]
      outer_zeta = self.dependencies()[outer_index]
    except IndexError:
      return None
    H = derivative(self.__F, inner_zeta, du = inner_u)
    H = ufl.algorithms.expand_derivatives(H)
    if len(H.integrals()) == 0:
      return None
    H = derivative(H, outer_zeta, du = TrialFunction(outer_zeta.function_space()))
    H = ufl.algorithms.expand_derivatives(H)
    if len(H.integrals()) == 0:
      return None
    if hermitian:
      H = adjoint(H)
    replace_map = dict(zip(self.dependencies(), deps))
    H = replace(H, replace_map)

#     return action(H, u)
    return assemble(action(H, u), form_compiler_parameters = self.__form_compiler_parameters)

  def jacobian_solve(self, u, deps, b, hermitian):
    J_solver = self.__J_solvers[0]
    if hasattr(J_solver, "solve_transpose"):
      J_hermitian = False
    else:
      J_hermitian = hermitian
    J_solver = self.__J_solvers[1 if J_hermitian else 0]
    
    if self.__J_mats[1 if J_hermitian else 0] is None:
      replace_map = dict(zip(self.dependencies(), deps))   
      J = replace(self.__J, replace_map)
      if J_hermitian:
        J = adjoint(J) 
      J = assemble(J, form_compiler_parameters = self.__form_compiler_parameters)
      [hbc.apply(J, b) for hbc in self.__hbcs]
      if self.__cache_jacobian:
        self.__J_mats[1 if J_hermitian else 0] = J
    else:
      J = self.__J_mats[1 if J_hermitian else 0]
      self.apply_hbcs(b)
    
    if J_solver is None:
      J_solver = _linear_solver(self.__linear_solver_parameters)
      J_solver.set_operator(J)
      self.__J_solvers[1 if J_hermitian else 0] = J_solver
    elif not self.__cache_jacobian:
      J_solver.set_operator(J)
    
    if hermitian and not J_hermitian:
      J_solver.solve_transpose(u.vector(), b)
    else:
      J_solver.solve(u.vector(), b)
    
    return
    
  def reset_jacobian_solve(self):
    self.__J_mats = [None, None]
    self.__J_solvers = [None, None]
    return
  
  def tangent_linear(self, m, tlm_map):
    x = self.x()
    if m == x:
      raise DACException("Invalid tangent-linear parameter")
      
    tlm_rhs = -derivative(self.__F, m, du = Constant(1.0, static = True))
    for dep in self.dependencies():
      if dep != x and dep in tlm_map:
        tlm_rhs -= derivative(self.__F, dep, du = tlm_map[dep])
    
    return EquationSolver(self.__J == tlm_rhs, tlm_map[x], self.__hbcs,
      solver_parameters = self.__linear_solver_parameters,
      form_compiler_parameters = self.__form_compiler_parameters,
      cache_jacobian = self.__cache_jacobian)

