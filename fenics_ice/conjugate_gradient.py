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

import numpy as np

def conjgrad(A, b, x, max_iter = None, tol=1e-10):


    max_iter = length(b) if isequal(max_iter, None)

    r = b - np.inner(A,x);
    p = r;
    rs_old = np.inner(r, r);

    for i in range(len(b)):
        Ap = np.inner(A, p);
        alpha = rs_old / np.inner(p, Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rs_new = np.inner(r, r);

        #Check stopping criterion
        if np.sqrt(rs_new) < tol:
            print ''' Norm of residual is less than tolerance (%1.2e) \n
                    Converged in %i iterations ''' % (tol, i)
            print i
            return x
        elif i > max_iter:
            print '''Maximum number of iterations reached (%i) \n
                    Converged in %i iterations ''' % (max_iter, i)

        p = r + (rs_new / rs_old) * p;
        rs_old = rs_new;

A = np.eye(3)
b = np.array([1.0, 2.0,3.0])
x0 = np.array([1.0, 2.0,3.0])

A = array([[4.,1.],[3.,1.]])
