"""
Decorator library - TODO should this be merged with something else?
"""

import logging

class count_calls:  # noqa: N801
    """
    Decorator to count function calls

    Log interval can be set, e.g. @count_calls(1000)
    """

    def __init__(self, interval=1):
        """Initialize counter, logger and logging interval"""
        self.n = 0
        self.interval = interval
        self.log = logging.getLogger("fenics_ice")

    def __call__(self, f):
        """Return wrapped function"""
        def wrapped(*args, **kwargs):
            self.n += 1
            if(self.n % self.interval == 0):
                self.log.info("%s call %s" % (f.__name__, self.n))
            return f(*args, **kwargs)

        return wrapped


def flag_errors(fn):
    """
    Catch errors from SLEPc/PETSc

    Copied from tlm_adjoint eigendecomposition.py. Not sure this
    will work as intended, because eps_error is defined in this
    decorator, whereas in tlm_adjoint it's in the main scope.
    """
    eps_error = [False]

    def wrapped_fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except:  # noqa: E722
            eps_error[0] = True
            raise
    return wrapped_fn
