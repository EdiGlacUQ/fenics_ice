#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function

from dolfin import Function, GenericMatrix as Matrix, GenericVector as Vector, \
  TrialFunction, action, adjoint, as_backend_type, derivative, \
  has_lu_solver_method, norm, parameters, replace
from dolfin_adjoint import KrylovSolver, LUSolver, assemble, solve
from dolfin_adjoint import adjglobals, adjlinalg, compatibility, solving
from petsc4py import PETSc

import copy
import libadjoint
import ufl
import unittest

__all__ = \
  [
    "AssignmentSolver",
    "CustomEquation",
    "DACException",
    "EquationSolver"
  ]

class DACException(Exception):
  pass

def next_block_id():
  id = getattr(next_block_id, "id", 0)
  next_block_id.id = id + 1
  return id

def assign_vector(x, y):
  x = as_backend_type(x).vec()
  if not isinstance(x, PETSc.Vec):
    x.zero()
    x.axpy(1.0, y)
    return
  y = as_backend_type(y).vec()
  if not isinstance(y, PETSc.Vec):
    x.zero()
    x.axpy(1.0, y)
    return
  x.setArray(y)
  return

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

    if self.__scale <> 1.0:
      F = Function(self.function_space())
      F.vector().axpy(self.__scale, b.data.vector())
      b = F.vector()
    else:
      b = b.data.vector()
    x = Function(self.function_space())
    self.__A._solve(x, b, self.__hermitian)
    v = adjlinalg.Vector(x)

    parameters["adjoint"]["stop_annotating"] = stop_annotating
    return v

class CustomBlock(libadjoint.Block):
  def __init__(self, A):
    name = "DAC %s %i" % (A.__class__.__name__, next_block_id())
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
    stop_annotating = parameters["adjoint"]["stop_annotating"]
    parameters["adjoint"]["stop_annotating"] = True

    self.__rhs.jacobian_solve(x, self.__values, b, hermitian)

    parameters["adjoint"]["stop_annotating"] = stop_annotating
    return

class CustomRHS(libadjoint.RHS):
  def __init__(self, eq, deps):
    self.__eq = eq
    self.__deps = copy.copy(deps)
    return

  def __call__(self, dependencies, values):
    stop_annotating = parameters["adjoint"]["stop_annotating"]
    parameters["adjoint"]["stop_annotating"] = True

    deps = copy.copy(self.__eq.dependencies())
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
    if not isinstance(contraction_vector.data, Function):
      return adjlinalg.Vector(None)

    stop_annotating = parameters["adjoint"]["stop_annotating"]
    parameters["adjoint"]["stop_annotating"] = True

    self.__eq.reset_jacobian()
    b = self.__eq.derivative_action([value.data for value in values], dependencies.index(variable), contraction_vector.data, hermitian)
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

  def derivative_assembly(self, dependencies, values, variable, hermitian):
    return JacobianMatrix(self, [value.data for value in values], hermitian)

  def jacobian_solve(self, *args, **kwargs):
    self.__eq.reset_jacobian()
    return self.__eq.jacobian_solve(*args, **kwargs)

class CustomEquation(object):
  def __init__(self, x, deps):
    """
    A custom equation. This provides an "escape hatch" via which arbitrary
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

    self.__A_block = CustomEquationMatrix(x.function_space()).block()
    self.__x = x
    self.__deps = copy.copy(deps)

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

  def _annotate(self):
    x = self.x()
    deps = self.dependencies()

    var = adjglobals.adj_variables[x]
    solving.register_initial_conditions([[x, var]], linear = False)
    var = adjglobals.adj_variables.next(x)

    dep_vars = [adjglobals.adj_variables[dep] for dep in deps]
    solving.register_initial_conditions(zip(deps, dep_vars), linear = False, var = var)

    rhs = CustomRHS(self, dep_vars)
    eqn = libadjoint.Equation(var, blocks = [self.__A_block], targets = [var], rhs = rhs)

    cs = adjglobals.adjointer.register_equation(eqn)
    solving.do_checkpoint(cs, var, rhs)

    # As in assignment.register_assign
    if parameters["adjoint"]["record_all"]:
      adjglobals.adjointer.record_variable(var, libadjoint.MemoryStorage(adjlinalg.Vector(x)))

    return

  def solve(self, annotate = None):
    """
    Solve the equation.

    Arguments:

    annotate  (Optional) Whether the equation should be recorded on the
              libadjoint tape. Overrides the dolfin-adjoint "stop_annotating"
              parameter.
    """

    if annotate is None:
      annotate = not parameters["adjoint"]["stop_annotating"]

    stop_annotating = parameters["adjoint"]["stop_annotating"]
    parameters["adjoint"]["stop_annotating"] = True

    self.forward_solve(self.x(), self.dependencies())
    if annotate:
      self._annotate()

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

  def set_bc_dofs(self, v, d):
    """
    Set the value of Dirichlet BC degrees of freedom in the Vector v to be equal
    to the corresponding degrees of freedom in the Vector d.
    """

    raise DACException("Method not overridden")

  def residual(self, r, deps):
    """
    Compute the equation residual. The libadjoint tape is disabled when this
    method is called.

    Arguments
      r     A Vector. The residual, which should be set by this method.
      deps  A list of Function objects defining the values of dependencies.
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

  def reset_jacobian(self):
    """
    Reset the system Jacobian. Can be used to clear caches used by
    derivative_action and jacobian_solve.
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

  def jacobian_solve(self, u, deps, b, hermitian, approximate_jacobian = False):
    """
    Solve a tangent linear or adjoint equation. The libadjoint tape is disabled
    when this method is called.

    Arguments:
      u          A Function. The solution, which should be set by this method.
      deps       A list of Function objects defining the values of dependencies.
      b          A Vector defining the right-hand-side.
      hermitian  If True then the equation solved is (dF/dx)^* u = b. Otherwise
                 the equation solved is dF/dx u = b.
      approximate_jacobian  (Optional) Whether it is acceptable to replace the
                            Jacobian with some approximation (e.g. for a Picard
                            solver).
    """

    raise DACException("Method not overridden")

class AssignmentSolver(CustomEquation):
  def __init__(self, x, y):
    CustomEquation.__init__(self, x, [x, y])
    return

  def apply_bcs(self, v):
    return

  def apply_hbcs(self, v):
    return

  def set_bc_dofs(self, v, d):
    return

  def residual(self, r, deps):
    if len(deps) <> 2:
      raise DACException("Expected exactly two dependencies")
    assign_vector(r, deps[0].vector())
    r.axpy(-1.0, deps[1].vector())
    return

  def forward_solve(self, x, deps):
    if len(deps) <> 2:
      raise DACException("Expected exactly two dependencies")
    x.assign(deps[1])
    return

  def derivative_action(self, deps, index, u, hermitian):
    if len(deps) <> 2:
      raise DACException("Expected exactly two dependencies")
    if index == 0:
      return u
    elif index == 1:
      F = Function(self.function_space())
      F.vector().axpy(-1.0, u.vector())
      return F
    else:
      return None

  def jacobian_solve(self, u, deps, b, hermitian, approximate_jacobian = False):
    assign_vector(u.vector(), b)
    return

class EquationSolver(CustomEquation):
  def __init__(self, *args, **kwargs):
    kwargs = copy.copy(kwargs)
    if "initial_guess" in kwargs:
      initial_guess = kwargs["initial_guess"]
      del(kwargs["initial_guess"])
    else:
      initial_guess = None
    if "cache_jacobian" in kwargs:
      cache_J = kwargs["cache_jacobian"]
      del(kwargs["cache_jacobian"])
    else:
      cache_J = False
    eq, x, bcs, J, _, _, form_compiler_parameters, solver_parameters = compatibility._extract_args(*args, **kwargs)
    if not J is None:
      raise DACException("Custom Jacobian not supported")

    lhs, rhs = eq.lhs, eq.rhs
    deps = set()
    for dep in ufl.algorithms.extract_coefficients(lhs):
      if isinstance(dep, Function):
        deps.add(dep)
    if not rhs == 0:
      for dep in ufl.algorithms.extract_coefficients(rhs):
        if isinstance(dep, Function):
          deps.add(dep)
    if not initial_guess is None:
      if initial_guess == x:
        raise DACException("Cannot use solution as an initial guess")
      deps.add(initial_guess)
    deps = list(deps)

    linear = isinstance(lhs, ufl.form.Form) and isinstance(rhs, ufl.form.Form)
    if linear:
      if x in deps:
        raise DACException("Non-linear dependency in linear equation")
      deps.insert(0, x)
      F = action(lhs, x) - rhs
      J = lhs
    else:
      if x in deps:
        deps.remove(x)
      deps.insert(0, x)
      F = lhs
      if not rhs == 0:
        F -= rhs
      J = derivative(F, x, du = TrialFunction(x.function_space()))

    hbcs = [compatibility.bc(bc) for bc in bcs]
    [hbc.homogenize() for hbc in hbcs]

    def update_parameters(old_parameters, new_parameters):
      for key, value in new_parameters.iteritems():
        if not key in old_parameters:
          old_parameters[key] = copy.deepcopy(value)
        elif isinstance(old_parameters[key], dict):
          if not isinstance(value, dict):
            raise DACException("Invalid solver parameter: %s" % key)
          update_parameters(old_parameters[key], value)
        else:
          old_parameters[key] = old_parameters[key].__class__(value)
      return
    if cache_J:
      linear_solver_parameters = {"lu_solver":{"reuse_factorization":True, "same_nonzero_pattern":True},
                                  "krylov_solver":{"preconditioner":{"structure":"same"}}}
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

    linear_solver = linear_solver_parameters.get("linear_solver", "default")
    if linear_solver in ["direct", "lu"]:
      linear_solver = "default"
    elif linear_solver == "iterative":
      linear_solver = "gmres"
    is_lu_linear_solver = linear_solver == "default" or has_lu_solver_method(linear_solver)

    CustomEquation.__init__(self, x, deps)
    self.__F = F
    self.__lhs, self.__rhs = lhs, rhs
    self.__bcs = copy.copy(bcs)
    self.__hbcs = hbcs
    self.__J = J
    self.__form_compiler_parameters = copy.deepcopy(form_compiler_parameters)
    self.__solver_parameters = copy.deepcopy(solver_parameters)
    self.__linear_solver_parameters = linear_solver_parameters
    self.__linear_solver = linear_solver
    self.__is_lu_linear_solver = is_lu_linear_solver
    self.__initial_guess = initial_guess
    self.__deps = deps
    self.__linear = linear

    self.__cache_J = cache_J
    self.reset_jacobian()

    return

  def apply_bcs(self, v):
    [bc.apply(v) for bc in self.__bcs]
    return

  def apply_hbcs(self, v):
    [bc.apply(v) for bc in self.__hbcs]
    return

  def set_bc_dofs(self, v, d):
    v_ = v.array()
    d_ = d.array()
    for bc in self.__bcs:
      i = bc.get_boundary_values().keys()
      v_[i] = d_[i]
    v.set_local(v_)
    v.apply("insert")

    return

  def residual(self, r, deps):
    replace_map = dict(zip(self.__deps, deps))
    lhs = replace(self.__lhs, replace_map)
    if self.__rhs == 0:
      assign_vector(r, assemble(lhs))
    else:
      assign_vector(r, assemble(action(lhs, replace_map[self.x()]) - replace(self.__rhs, replace_map)))
    self.apply_hbcs(r)
    return

  def forward_solve(self, x, deps):
    replace_map = dict(zip(self.__deps, deps))

    if not self.__initial_guess is None:
      x.assign(replace_map[self.__initial_guess])
    replace_map[self.x()] = x

    lhs = replace(self.__lhs, replace_map)
    rhs = 0 if self.__rhs == 0 else replace(self.__rhs, replace_map)
    J = replace(self.__J, replace_map)

    solve(lhs == rhs, x, self.__bcs, J = J, form_compiler_parameters = self.__form_compiler_parameters, solver_parameters = self.__solver_parameters)

    return

  def reset_jacobian(self):
    if self.__cache_J:
      self.__J_cache = {}
      self.__J_mats = [None, None]
    self.__J_solvers = [None, None]
    return

  def derivative_action(self, deps, index, u, hermitian):
    def J_form():
      replace_map = dict(zip(self.__deps, deps))
      zeta = self.__deps[index]
      J = derivative(self.__F, zeta, du = TrialFunction(zeta.function_space()))
      J = ufl.algorithms.expand_derivatives(J)
      if len(J.integrals()) == 0:
#         return ufl.zero()
        if hermitian:
          return Function(deps[index].function_space()).vector()
        else:
          return Function(self.function_space()).vector()
      if hermitian:
        J = adjoint(J)
      J = replace(J, replace_map)
      return J

    if self.__cache_J:
      key = (index, hermitian)
      J = self.__J_cache.get(key, None)
      if J is None:
        J = J_form()
        if not isinstance(J, ufl.form.Form):
          self.__J_cache[key] = J
        else:
          J = self.__J_cache[key] = assemble(J, form_compiler_parameters = self.__form_compiler_parameters)
      if isinstance(J, Matrix):
        return J * u.vector()
      else:
        return J
    else:
      J = J_form()
      if not isinstance(J, ufl.form.Form):
        return J
#       return action(J, u)
      return assemble(action(J, u), form_compiler_parameters = self.__form_compiler_parameters)

  def jacobian_solve(self, u, deps, b, hermitian, approximate_jacobian = False):
    J_hermitian = hermitian
    J_solver = self.__J_solvers[1 if J_hermitian else 0]

    if J_solver is None or not self.__cache_J:
      replace_map = dict(zip(self.__deps, deps))
      J = replace(self.__J, replace_map)
      if J_hermitian:
        J = adjoint(J)
      J = assemble(J, form_compiler_parameters = self.__form_compiler_parameters)
      [hbc.apply(J, b) for hbc in self.__hbcs]
      if self.__cache_J:
        self.__J_mats[1 if J_hermitian else 0] = J
    else:
      J = self.__J_mats[1 if J_hermitian else 0]
      self.apply_hbcs(b)

    if J_solver is None:
      if self.__is_lu_linear_solver:
        J_solver = LUSolver(self.__linear_solver)
        J_solver.parameters.update(self.__linear_solver_parameters.get("lu_solver", {}))
      else:
        J_solver = KrylovSolver(self.__linear_solver, self.__linear_solver_parameters.get("preconditioner", "none"))
        J_solver.parameters.update(self.__linear_solver_parameters.get("krylov_solver", {}))
      self.__J_solvers[1 if J_hermitian else 0] = J_solver

    if self.__is_lu_linear_solver:
      J_solver.set_operator(J)
      J_solver.solve(u.vector(), b)
    else:
      J_solver.set_operator(J)
      J_solver.solve(u.vector(), b)

    return

class dolfin_adjoint_custom_tests(unittest.TestCase):
  def testEquationSolver(self):
    from dolfin import DirichletBC, Expression, FunctionSpace, TestFunction, \
      UnitIntervalMesh, dx, grad, inner
    from dolfin_adjoint import Constant
    from dolfin_adjoint import FunctionControl, Functional, adj_checkpointing, \
      adj_check_checkpoints, adj_inc_timestep, adj_start_timestep, adj_reset, \
      compute_gradient, taylor_test

    adj_reset()
    parameters["adjoint"]["stop_annotating"] = False
    n_steps = 20
    adj_checkpointing(strategy = "multistage", steps = n_steps, snaps_on_disk = 4, snaps_in_ram = 2, verbose = True)

    mesh = UnitIntervalMesh(100)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    T_0 = Function(space, name = "T_0")
    T_0.interpolate(Expression("sin(pi * x[0]) + sin(10.0 * pi * x[0])", element = space.ufl_element()))
    dt = Constant(0.01)
    kappa = Constant(0.01)

    def forward(T_0):
      T_n = Function(space, name = "T_n")
      T_np1 = Function(space, name = "T_np1")

      T_n.assign(T_0, annotate = False)
      eq = EquationSolver(inner(test, trial) * dx + dt * inner(grad(test), kappa * grad(trial)) * dx == inner(test, T_n) * dx,
             T_np1, DirichletBC(space, 0.0, "on_boundary"), solver_parameters = {"linear_solver":"gmres",
                                                                                 "krylov_solver":{"relative_tolerance":1.0e-14,
                                                                                                  "absolute_tolerance":1.0e-16}})
      adj_start_timestep()
      for n in xrange(n_steps):
        eq.solve()
        T_n.assign(T_np1)
        adj_inc_timestep()

      return T_n

    T_N = forward(T_0)
    parameters["adjoint"]["stop_annotating"] = True

    adj_check_checkpoints()

    J_val = assemble(inner(T_N, T_N) * dx)
    self.assertAlmostEquals(J_val, 4.9163656210568579e-01)

    J = Functional(inner(T_N, T_N) * dx)
    control = FunctionControl("T_n")
    dJ = compute_gradient(J, control, forget = False)
    def J_test(T_0):
      T_N = forward(T_0)
      return assemble(inner(T_N, T_N) * dx)
    minconv = taylor_test(J_test, control, J_val, dJ)
    self.assertGreaterEqual(minconv, 1.99)

    adj_reset()
    return

  def testAssignmentSolver(self):
    from dolfin import FunctionSpace, TestFunction, UnitIntervalMesh, dx, inner
    from dolfin_adjoint import Constant
    from dolfin_adjoint import FunctionControl, Functional, adj_reset, \
      compute_gradient, replay_dolfin, taylor_test

    adj_reset()
    parameters["adjoint"]["stop_annotating"] = False

    mesh = UnitIntervalMesh(100)
    space = FunctionSpace(mesh, "DG", 0)
    test, trial = TestFunction(space), TrialFunction(space)
    x = Function(space, name = "x")
    x.assign(Constant(1.0))

    def forward(x):
      y = [Function(space, name = "y_%i" % i) for i in xrange(9)]
      z = Function(space, name = "z")

      AssignmentSolver(y[0], x).solve()
      for i in xrange(0, len(y) - 1):
        AssignmentSolver(y[i + 1], y[i]).solve()
      solve(inner(test, trial) * dx == inner(test, y[-1] * y[-1]) * dx, z)
      return z

    z = forward(x)
    parameters["adjoint"]["stop_annotating"] = True

    self.assertTrue(replay_dolfin())

    J_val = assemble(inner(z, z) * dx)
    self.assertAlmostEquals(J_val, 1.0)

    J = Functional(inner(z, z) * dx)
    control = FunctionControl(x)
    dJ = compute_gradient(J, control, forget = False)
    def J_test(x):
      z = forward(x)
      return assemble(inner(z, z) * dx)
    minconv = taylor_test(J_test, control, J_val, dJ)
    self.assertGreaterEqual(minconv, 2.0)

    adj_reset()
    return

if __name__ == "__main__":
  import numpy
  numpy.random.seed(1201)
  unittest.main()
