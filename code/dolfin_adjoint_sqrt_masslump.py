#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import TestFunction, dx
from dolfin_adjoint import Function, assemble
from dolfin_adjoint_custom import CustomEquation, DACException

import numpy

__all__ = \
  [
    "SqrtMasslumpEquation",
    "sqrt_masslump_action",
    "sqrt_inv_masslump_action"
  ]

class SqrtMasslumpEquation(CustomEquation):
  """
  Defines an equation
     M_L^1/2 \\tilde{x} = \\tilde{y},
  where \\tilde{x} is a vector of coefficients for a P0 or P1 Function, and M_L
  is a lumped mass matrix with elements given by
    M_{L,i,j} = \delta_{i,j} \int_\Omega \phi_i.

  Usage: If x and y are P0 or P1 functions in the same space, first instantiate
  the equation
    y = Function(x.function_space())
    eq = SqrtMasslumpEquation(y, x)
  If x_0 is a Function in the same space as x storing the value for the control,
  initialise y and x (with the libadjoint tape enabled for the second line)
    eq.sqrt_masslump_action(x_0, y = y)
    eq.solve()
  Optimisation is performed using y as the control
    J = Function(...)
    control = Control(y)
    J_hat = ReducedFunctional(J, control)
    y_opt = minimize(J_hat, ...).
  To optain the associated value for the control
    x_opt = eq.sqrt_inv_masslump_action(y_opt)
  or, if eq is no longer available
    x_opt = sqrt_inv_masslump_action(y_opt)
  """

  def __init__(self, y, x):
    CustomEquation.__init__(self, x, [x, y])
    self.__M_L_sqrt = numpy.sqrt(assemble(TestFunction(x.function_space()) * dx).array())
    return

  def apply_bcs(self, v):
    return

  def apply_hbcs(self, v):
    return

  def forward_solve(self, x, deps):
    self.jacobian_solve(x, deps, deps[1].vector(), hermitian = False)
    return

  def derivative_action(self, deps, index, u, hermitian):
    if index == 0:
      F = Function(self.function_space())
      F.vector().set_local(u.vector().array() * self.__M_L_sqrt)
      F.vector().apply("insert")
      return F
    elif index == 1:
      F = Function(self.function_space())
      F.vector().axpy(-1.0, u.vector())
      return F
    else:
      return None

  def second_derivative_action(self, deps, inner_index, inner_u, outer_index, u, hermitian):
    return None

  def jacobian_solve(self, u, deps, b, hermitian):
    u.vector().set_local(b.array() / self.__M_L_sqrt)
    u.vector().apply("insert")
    return

  def sqrt_masslump_action(self, x, y = None):
    if y is None:
      y = Function(self.function_space())
    y.vector().set_local(x.vector().array() * self.__M_L_sqrt)
    y.vector().apply("insert")
    return y

  def sqrt_inv_masslump_action(self, y, x = None):
    if x is None:
      x = Function(self.function_space())
    x.vector().set_local(y.vector().array() / self.__M_L_sqrt)
    x.vector().apply("insert")
    return x

  def tangent_linear(self, m, tlm_map):
    x, y = self.dependencies()
    if m in [x, y]:
      raise DACException("Invalid tangent-linear parameter")
    elif not y in tlm_map:
      return NullSolver(tlm_map[x])
    else:
      return SqrtMasslumpEquation(tlm_map[y], tlm_map[x])

def sqrt_masslump_action(x, y = None):
  if y is None:
    y = Function(x.function_space())
  y.vector().set_local(x.vector().array() * numpy.sqrt(assemble(TestFunction(x.function_space()) * dx).array()))
  y.vector().apply("insert")
  return y

def sqrt_inv_masslump_action(y, x = None):
  if x is None:
    x = Function(y.function_space())
  x.vector().set_local(y.vector().array() / numpy.sqrt(assemble(TestFunction(y.function_space()) * dx).array()))
  x.vector().apply("insert")
  return x
