from fenics import *
from tlm_adjoint.fenics import *
from fenics_ice.interpolation import *

import numpy as np
import os
import pytest
import tlm_adjoint
import sys

sys.path.insert(0, os.path.join(os.path.dirname(tlm_adjoint.__file__),
                                os.path.pardir))
from tests.fenics.test_base import *  # noqa: E402
del sys.path[0]


@pytest.mark.parametrize("N_x, N_y, N_z", [(5, 5, 2),
                                           (5, 5, 5)])
@seed_test
def test_Interpolation(setup_test, test_leaks, test_ghost_modes,
                       N_x, N_y, N_z):
    mesh = UnitCubeMesh(N_x, N_y, N_z)
    X = SpatialCoordinate(mesh)
    z_space = FunctionSpace(mesh, "Lagrange", 3)
    if DEFAULT_COMM.size > 1:
        y_space = FunctionSpace(mesh, "Discontinuous Lagrange", 3)
    x_space = FunctionSpace(mesh, "Lagrange", 2)

    # Test optimization: Use to cache the interpolation matrix
    P = [None]

    def forward(z):
        if DEFAULT_COMM.size > 1:
            y = Function(y_space, name="y")
            LocalProjection(y, z).solve()
        else:
            y = z

        x = Function(x_space, name="x")
        eq = Interpolation(x, y, P=P[0], tolerance=1.0e-15)
        eq.solve()
        P[0] = eq._B[0]._A._P

        J = Functional(name="J")
        J.assign((dot(x + Constant(1.0), x + Constant(1.0)) ** 2) * dx)
        return x, J

    z = Function(z_space, name="z", static=True)
    interpolate_expression(z,
                           sin(pi * X[0]) * sin(2.0 * pi * X[1]) * exp(X[2]))
    start_manager()
    x, J = forward(z)
    stop_manager()

    x_ref = Function(x_space, name="x_ref")
    x_ref.interpolate(z)

    x_error = Function(x_space, name="x_error")
    var_assign(x_error, x_ref)
    var_axpy(x_error, -1.0, x)

    x_error_norm = var_linf_norm(x_error)
    info(f"Error norm = {x_error_norm:.16e}")
    assert x_error_norm < 1.0e-13

    J_val = J.value

    dJ = compute_gradient(J, z)

    def forward_J(z):
        return forward(z)[1]

    min_order = taylor_test(forward_J, z, J_val=J_val, dJ=dJ, seed=1.0e-4)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, z, J_val=J_val, ddJ=ddJ, seed=1.0e-3)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, z, tlm_order=1, seed=1.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, z, adjoint_order=1,
                                        seed=1.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, z, adjoint_order=2,
                                        seed=1.0e-4)
    assert min_order > 1.99


@pytest.mark.parametrize("N_x, N_y, N_z", [(2, 2, 2),
                                           (5, 5, 5)])
@pytest.mark.parametrize("c", [-1.5, 1.5])
@seed_test
def test_PointInterpolation(setup_test, test_leaks, test_ghost_modes,
                            N_x, N_y, N_z,
                            c):
    mesh = UnitCubeMesh(N_x, N_y, N_z)
    X = SpatialCoordinate(mesh)
    z_space = FunctionSpace(mesh, "Lagrange", 3)
    if DEFAULT_COMM.size > 1:
        y_space = FunctionSpace(mesh, "Discontinuous Lagrange", 3)
    X_coords = np.array([[0.1, 0.1, 0.1],
                         [0.2, 0.3, 0.4],
                         [0.9, 0.8, 0.7],
                         [0.4, 0.2, 0.3]], dtype=backend_RealType)

    # Test optimization: Use to cache the interpolation matrix
    P = [None]

    def forward(z):
        if DEFAULT_COMM.size > 1:
            y = Function(y_space, name="y")
            LocalProjection(y, z).solve()
        else:
            y = z

        X_vals = [Constant(name=f"x_{i:d}")
                  for i in range(X_coords.shape[0])]
        eq = PointInterpolation(X_vals, y, X_coords, P=P[0])
        eq.solve()
        P[0] = eq._P

        J = Functional(name="J")
        for x in X_vals:
            term = Constant()
            ExprInterpolation(term, x ** 3).solve()
            J.addto(term)
        return X_vals, J

    z = Function(z_space, name="z", static=True)
    interpolate_expression(z, pow(X[0], 3) - 1.5 * X[0] * X[1] + c)

    start_manager()
    X_vals, J = forward(z)
    stop_manager()

    def x_ref(x):
        return x[0] ** 3 - 1.5 * x[0] * x[1] + c

    x_error_norm = 0.0
    assert len(X_vals) == len(X_coords)
    for x, x_coord in zip(X_vals, X_coords):
        x_error_norm = max(x_error_norm,
                           abs(var_scalar_value(x) - x_ref(x_coord)))
    info(f"Error norm = {x_error_norm:.16e}")
    assert x_error_norm < 1.0e-13

    J_val = J.value

    dJ = compute_gradient(J, z)

    def forward_J(z):
        return forward(z)[1]

    min_order = taylor_test(forward_J, z, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, z, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, z, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, z, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, z, adjoint_order=2)
    assert min_order > 1.99
