from fenics import *
from tlm_adjoint.fenics import *

stop_manager()


def _Vector_inner(self, /, other):
    if hasattr(self, "_tlm_adjoint__function") \
            and hasattr(other, "_tlm_adjoint__function"):
        check_space_types_conjugate_dual(self._tlm_adjoint__function,
                                         other._tlm_adjoint__function)
    return _Vector__inner_orig(self, other)


from tlm_adjoint.fenics.backend import cpp_PETScVector  # noqa: E402
_Vector__inner_orig = cpp_PETScVector.inner
cpp_PETScVector.inner = _Vector_inner
del _Vector_inner, cpp_PETScVector
