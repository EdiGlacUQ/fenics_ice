from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint_sqrt_masslump import *
from dolfin_adjoint_custom import EquationSolver
import moola
import numpy as np
import ufl
import os
import timeit
import time
from IPython import embed
import eigendecomposition
import eigenfunc



class ssa_solver:

    def __init__(self, model):
        # Enable aggressive compiler options
        parameters["form_compiler"]["optimize"] = False
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math -march=native"


        self.model = model
        self.param = model.param

        #Fields
        self.bed = model.bed
        self.H_np = model.H_np
        self.H_s = model.H_s
        self.H = model.H
        #self.surf = model.surf
        self.beta = model.beta
        self.beta_bgd = model.beta_bgd
        self.mask = model.mask
        self.alpha = model.alpha
        self.bmelt = model.bmelt
        self.latbc = model.latbc

        #Facet normals
        self.nm = model.nm

        #Save observations for inversions
        try:
            self.u_obs = model.u_obs
            self.v_obs = model.v_obs
            self.u_std = model.u_std
            self.v_std = model.v_std
        except:
            pass

        #Mesh/Function Spaces
        self.mesh = model.mesh
        self.V = model.V
        self.Q = model.Q
        self.M = model.M
        self.RT = model.RT

        #Trial/Test Functions
        self.U = Function(self.V, name = "U")
        self.U_np = Function(self.V, name = "U_np")
        self.Phi = TestFunction(self.V)
        self.Ksi = TestFunction(self.M)

        self.trial_H = TrialFunction(self.M)
        self.H_nps = Function(self.M)

        #Cells
        self.cf = model.cf
        self.OMEGA_DEF = model.OMEGA_DEF
        self.OMEGA_ICE_FLT = model.OMEGA_ICE_FLT
        self.OMEGA_ICE_GND = model.OMEGA_ICE_GND
        self.OMEGA_ICE_FLT_OBS = model.OMEGA_ICE_FLT_OBS
        self.OMEGA_ICE_GND_OBS = model.OMEGA_ICE_GND_OBS

        #Facets
        self.ff = model.ff
        self.GAMMA_DEF = model.GAMMA_DEF
        self.GAMMA_LAT = model.GAMMA_LAT
        self.GAMMA_TMN = model.GAMMA_TMN      #Value at ice terminus
        self.GAMMA_NF = model.GAMMA_NF


        #Measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.cf)
        self.dS = Measure('dS', domain=self.mesh, subdomain_data=self.ff)
        self.ds = dolfin.ds

        self.dIce = self.dx
        self.dIce_flt = self.dx(self.OMEGA_ICE_FLT) + self.dx(self.OMEGA_ICE_FLT_OBS)
        self.dIce_gnd = self.dx(self.OMEGA_ICE_GND) + self.dx(self.OMEGA_ICE_GND_OBS)

        self.dObs = self.dx(self.OMEGA_ICE_FLT_OBS) + self.dx(self.OMEGA_ICE_GND_OBS)
        self.dObs_gnd = self.dx(self.OMEGA_ICE_FLT_OBS)
        self.dObs_flt = self.dx(self.OMEGA_ICE_GND_OBS)

        self.dt = Constant(self.param['dt'])



    def def_mom_eq(self):

        #Simplify accessing fields and parameters
        bed = self.bed
        H = self.H
        #surf = self.surf
        mask = self.mask
        alpha = self.alpha


        rhoi = self.param['rhoi']
        rhow = self.param['rhow']
        delta = 1.0 - rhoi/rhow
        g = self.param['g']
        n = self.param['n']
        tol = self.param['tol']
        dIce = self.dIce
        ds = self.ds

        #Vector components of trial function
        u, v = split(self.U)

        #Vector components of test function
        Phi = self.Phi
        Phi_x, Phi_y = split(Phi)

        #Derivatives
        u_x, u_y = u.dx(0), u.dx(1)
        v_x, v_y = v.dx(0), v.dx(1)

        #Viscosity
        U_marker = Function(self.U.function_space(), name = "%s_marker" % self.U.name())
        nu = self.viscosity(U_marker)

        #Sliding law
        #B2 = exp(alpha)
        B2 = alpha*alpha

        #Switch parameters
        H_s = -rhow/rhoi * bed
        fl_ex = ufl.operators.Conditional(H <= H_s, Constant(1.0), Constant(0.0))

        #Driving stress quantities
        F = (1 - fl_ex) * 0.5*rhoi*g*H**2 + \
            (fl_ex) * 0.5*rhoi*g*(delta*H**2 + (1-delta)*H_s**2 )

        W = (1 - fl_ex) * rhoi*g*H + \
            (fl_ex) * rhoi*g*H_s

        draft = (fl_ex) * (rhoi / rhow) * H

        #Terminating margin boundary condition
        sigma_n = 0.5 * rhoi * g * ((H ** 2) - (rhow / rhoi) * (draft ** 2)) - F

        self.mom_F = (
                #Membrance Stresses
                -inner(grad(Phi_x), H * nu * as_vector([4 * u_x + 2 * v_y, u_y + v_x])) * self.dIce
                - inner(grad(Phi_y), H * nu * as_vector([u_y + v_x, 4 * v_y + 2 * u_x])) * self.dIce

                #Basal Drag
                - inner(Phi, (1.0 - fl_ex) * B2 * as_vector([u,v])) * self.dIce

                #Driving Stress
                + ( div(Phi)*F - inner(grad(bed),W*Phi) ) * self.dIce

                #Boundary condition
                + inner(Phi * sigma_n, self.nm) * self.ds )

        self.mom_Jac_p = replace(derivative(self.mom_F, self.U), {U_marker:self.U})
        self.mom_F = replace(self.mom_F, {U_marker:self.U})
        self.mom_Jac = derivative(self.mom_F, self.U)


    def solve_mom_eq(self, annotate_flag=True):
        #Dirichlet Boundary Conditons: Zero flow

        bc0 = DirichletBC(self.V, self.latbc, self.ff, self.GAMMA_LAT)
        bc1 = DirichletBC(self.V, (0.0, 0.0), self.ff, self.GAMMA_NF)
        self.bcs = [bc0, bc1]


        #Non zero initial perturbation
        #self.init_guess()
        #parameters['krylov_solver']['nonzero_initial_guess'] = True
        t0 = time.time()


        MomentumSolver(self.mom_F == 0, self.U, bcs = self.bcs, J_p=self.mom_Jac_p, picard_params = self.param['picard_params'], solver_parameters = self.param['newton_params']).solve(annotate=annotate_flag)
        t1 = time.time()
        print "Time for solve: ", t1-t0


    def def_thickadv_eq(self):
        U = self.U
        U_np = self.U_np
        Ksi = self.Ksi
        trial_H = self.trial_H
        H_np = self.H_np
        H_s = self.H_s
        H = self.H
        dt = self.dt
        nm = self.nm
        dIce = self.dIce
        ds = self.ds
        dS = self.dS

        self.thickadv = (inner(Ksi, ((trial_H - H_np) / dt)) * dIce
        - inner(grad(Ksi), U_np * 0.5 * (trial_H + H_np)) * dIce
        + inner(jump(Ksi), jump(0.5 * (dot(U_np, nm) + abs(dot(U_np, nm))) * 0.5 * (trial_H + H_np))) * dS
        + inner(Ksi, dot(U_np * 0.5 * (trial_H + H_np), nm)) * ds)

        self.thickadv_split = replace(self.thickadv, {U_np:0.5 * (self.U + self.U_np)})

        bc0 = DirichletBC(self.M, self.H, self.ff, self.GAMMA_LAT)
        bc1 = DirichletBC(self.M, (0.0), self.ff, self.GAMMA_TMN)
        self.H_bcs = [bc0, bc1]

    def solve_thickadv_eq(self):

        H_s = self.H_s
        a, L = lhs(self.thickadv), rhs(self.thickadv)
        solve(a==L,H_s,bcs = self.H_bcs)

    def solve_thickadv_split_eq(self):
        H_nps = self.H_nps
        a, L = lhs(self.thickadv_split), rhs(self.thickadv_split)
        solve(a==L,H_nps, bcs = self.H_bcs)

    def timestep(self, save = 1, adjoint_flag=1, outdir='./'):
        U = self.U
        U_np = self.U_np
        H = self.H
        H_np = self.H_np
        H_s = self.H_s
        H_nps = self.H_nps
        self.save_H_init(H)

        n_steps = self.param['n_steps']
        dt = self.dt

        if save:
            hdf_hts = HDF5File(self.mesh.mpi_comm(), os.path.join(outdir, 'H_ts.h5'), 'w')
            hdf_uts = HDF5File(self.mesh.mpi_comm(), os.path.join(outdir, 'U_ts.h5'), 'w')

            pvd_hts = File(os.path.join(outdir, "H_ts.pvd"), "compressed")
            pvd_uts = File(os.path.join(outdir, "U_ts.pvd"), "compressed")

            hdf_hts.write(H_np, 'H', 0.0)
            hdf_uts.write(U_np, 'U', 0.0)

            pvd_hts << (H_np, 0.0)
            pvd_uts << (U_np, 0.0)




        self.def_thickadv_eq()
        self.def_mom_eq()
        self.solve_mom_eq()
        U_np.assign(U)

        t=0.0

        if adjoint_flag:
            adj_start_timestep()

        for n in xrange(n_steps):
            begin("Starting timestep %i of %i, time = %.16e a" % (n + 1, n_steps, t))

            # Solve
            self.solve_thickadv_eq()
            self.solve_mom_eq()
            self.solve_thickadv_split_eq()

            U_np.assign(U)
            H_np.assign(H_nps)

            #Increment time
            n += 1
            t = n * float(dt)

            #Record
            if save:
                hdf_hts.write(H_np, 'H', t)
                hdf_uts.write(U_np, 'U', t)

                pvd_hts << (H_np, t)
                pvd_uts << (U_np, t)

            if adjoint_flag:
                end()
                adj_inc_timestep()

    def inversion(self, cntrl2):

        #Record value of functional during minimization
        self.F_vals = []; #Initialize array

        #Callback function during minimization storing cost function value
        def derivative_cb(j, dj, m):
            self.F_vals.append(j)
            print "j = %f" % (j)

        #Initial equation definition and solve
        self.def_mom_eq()
        self.solve_mom_eq()

        #Set up cost functional
        self.set_J_inv(verbose=True)
        J = Functional(self.J_inv)


        for j in range(4):
            cntrl = cntrl2[j % 2]
            #Control parameters and functional problem
            control = [Control(x) for x in cntrl] if type(cntrl) is list else Control(cntrl)
            rf = ReducedFunctional(J, control, derivative_cb_post = derivative_cb)


            if type(cntrl) is list:
                controlm = moola.DolfinPrimalVectorSet([moola.DolfinPrimalVector(x) for x in cntrl])
            else:
                controlm = moola.DolfinPrimalVector(cntrl)

            # if type(cntrl) is list:
            #     Hscale = []
            #     for c in cntrl:
            #         t0 = time.time()
            #         self.set_hessian_action(c)
            #         A = eigenfunc.HessWrapper(self.ddJ,c)
            #         [lam,v] = eigenfunc.eigens(A,k=10,n_iter=2)
            #         #ddJw = ddJ_wrapper(self.ddJ,c)
            #         #lam,v = eigendecomposition.eig(ddJw.ddJ_F.vector().local_size(), ddJw.apply, hermitian = True, N_eigenvalues = 1)
            #         Hscale.append(lam[0])
            #         t1 = time.time()
            #         print "Time for solve: {0} ".format(t1-t0)
            #         self.solve_mom_eq()

            def Hinit(x):
                """ Returns the primal representation. """
                #events.increment("Dual -> primal map")

                print(np.median(np.abs(x[0].array()))/np.median(np.abs(x[1].array())))
                vscale = [2.3e+08,10000.0]
                y = x.copy()
                for v,s in zip(y.vector_list,vscale):
                    v.data.vector().set_local(v.array()/s)

                if x.riesz_map.inner_product == "l2":
                    return moola.DolfinPrimalVectorSet([vec.primal() for vec in y.vector_list],
                    riesz_map = y.riesz_map)
                else:
                    primal_vecs = zeros(len(y), dtype = "object")
                    primal_vecs[:] = [v.primal() for v in y.vector_list]
                    return moola.DolfinPrimalVectorSet(y.riesz_map.riesz_inv * primal_vecs,
                    riesz_map = y.riesz_map)

            problem = MoolaOptimizationProblem(rf)
            solver = moola.BFGS(problem, controlm, options={'jtol': 1e-4,
                                                   'gtol': 1e-9,
                                                   'line_search_options' : {"ftol": 1e-4, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1},
                                                   'Hinit': Hinit if type(cntrl) is list else 'default',
                                                   'maxiter': self.param['inv_options']['maxiter'],
                                                   'mem_lim': 10})

            sol = solver.solve()
            opt_var = sol['control']

            if type(cntrl) is list:
                map(lambda x: x[0].vector().set_local(x[1].array()), zip(cntrl,opt_var))
            else:
                cntrl.vector().set_local(opt_var.array())




        #Scipy Optimization routine
        #opt_var = minimize(rf, method = 'L-BFGS-B', options = self.param['inv_options'])
        #map(lambda x: x[0].assign(x[1]), zip(cntrl,opt_var)) if type(cntrl) is list else cntrl.assign(opt_var)

        #Re-compute velocities with inversion results
        adj_reset()
        parameters["adjoint"]["stop_annotating"] = False
        self.solve_mom_eq()
        parameters["adjoint"]["stop_annotating"] = True

        #Print out inversion results/parameter values
        self.set_J_inv(verbose = True)

    def epsilon(self, U):
        """
        return the strain-rate tensor of self.U.
        """
        epsdot = sym(grad(U))
        return epsdot

    def effective_strain_rate(self, U):
        """
        return the effective strain rate squared.
        """
        eps_rp = self.param['eps_rp']

        eps = self.epsilon(U)
        exx = eps[0,0]
        eyy = eps[1,1]
        exy = eps[0,1]

        # Second invariant of the strain rate tensor squared
        eps_2 = (exx**2 + eyy**2 + exx*eyy + (exy)**2 + eps_rp**2)

        return eps_2

    def viscosity(self,U):
        B = self.beta*self.beta
        n = self.param['n']

        eps_2 = self.effective_strain_rate(U)
        nu = 0.5 * B * eps_2**((1.0-n)/(2.0*n))

        return nu

    def init_guess(self):
        #Simplify accessing fields and parameters
        bed = self.bed
        H = self.H
        alpha = self.alpha

        rhoi = self.param['rhoi']
        rhow = self.param['rhow']
        delta = 1.0 - rhoi/rhow
        g = self.param['g']
        n = self.param['n']
        tol = self.param['tol']

        B2 = alpha*alpha
        H_s = -rhow/rhoi * bed
        fl_ex = ufl.operators.Conditional(H <= H_s, 1.0, 0.0)

        s = project((1-fl_ex) * (bed + H),self.Q)
        grads = as_vector([s.dx(0), s.dx(1)])
        U_ = project((1-fl_ex)*(rhoi*g*H*grads)/B2, self.V)

        self.U.assign(U_)

        vtkfile = File('U_init.pvd')
        vtkfile << self.U

    def set_J_inv(self, verbose=False):

        u, v = split(self.U)
        u_obs = self.u_obs
        v_obs = self.v_obs
        u_std = self.u_std
        v_std = self.v_std

        alpha = self.alpha
        beta = self.beta
        beta_bgd = self.beta_bgd

        dIce = self.dIce
        ds = self.ds
        nm = self.nm

        gamma_a = self.param['rc_inv'][0]
        lambda_a = self.param['rc_inv'][1]
        lambda_b = self.param['rc_inv'][2]
        delta_a = self.param['rc_inv'][3]
        delta_b = self.param['rc_inv'][4]

        J_ls = gamma_a*(u_std**(-2.0)*(u-u_obs)**2.0 + v_std**(-2.0)*(v-v_obs)**2.0)*self.dObs


        grad_alpha = grad(alpha)
        grad_alpha_ = project(grad_alpha, self.RT)
        lap_alpha = div(grad_alpha_)

        betadiff = beta-beta_bgd

        grad_betadiff = grad(betadiff)
        grad_betadiff_ = project(grad_betadiff, self.RT)
        lap_beta = div(grad_betadiff_)

        reg_a = lambda_a * alpha - delta_a*lap_alpha
        reg_b = lambda_b * betadiff - delta_b*lap_beta

        J_reg_alpha = inner(reg_a,reg_a)*dIce
        J_reg_beta = inner(reg_b,reg_b)*dIce

        J = J_ls + J_reg_alpha + J_reg_beta

        self.J_inv = J


        if verbose:
            #Print out results
            J1 = assemble(J)
            J2 =  assemble(J_ls)
            J3 = assemble(J_reg_alpha)
            J4 = assemble(J_reg_beta)


            print 'Inversion Details'
            print 'lambda_a: %.2e' % lambda_a
            print 'lambda_b: %.2e' % lambda_b
            print 'delta_a: %.2e' % delta_a
            print 'delta_b: %.2e' % delta_b
            print 'J: %.2e' % J1
            print 'J_ls: %.2e' % J2
            print 'J_reg: %.2e' % sum([J3,J4])
            print 'J_reg_alpha: %.2e' % J3
            print 'J_reg_beta: %.2e' % J4
            print 'J_reg/J_cst: %.2e' % ((J3+J4)/(J2))

    def set_J_vaf(self, verbose=False):
        H = self.H_nps
        #B stands in for self.bed, which leads to a taping error
        B = Function(self.M)
        B.assign(self.bed, annotate=False)
        rhoi = Constant(self.param['rhoi'])
        rhow = Constant(self.param['rhow'])
        dIce = self.dIce
        dIce_gnd = self.dIce_gnd
        dt = self.dt

        b_ex = conditional(B < 0.0, 1.0, 0.0)

        HAF = b_ex * (H + rhow/rhoi*B) + (1-b_ex)*(H)

        self.J_vaf = HAF * dIce_gnd

    def comp_dJ_vaf(self, cntrl):
        J = Functional(self.J_vaf)
        control = Control(cntrl)
        dJ = compute_gradient(J, control, forget = False)
        self.dJ_vaf = dJ

    def comp_dJ_inv(self, cntrl):
        J = Functional(self.J_inv)
        control = [Control(x) for x in cntrl] if type(cntrl) is list else Control(cntrl)
        dJ = compute_gradient(J, control, forget = False)
        self.dJ_inv = dJ


    def set_hessian_action(self, cntrl):
        J = Functional(self.J_inv)
        cc = [Control(x) for x in cntrl] if type(cntrl) is list else Control(cntrl)
        self.ddJ = hessian(J,cc)

    def taylor_ver_inv(self,alpha_in):
        self.alpha = alpha_in
        self.def_mom_eq()
        self.solve_mom_eq()
        self.set_J_inv()
        return assemble(self.J_inv)

    def save_H_init(self,H):
        self.H_init = H

    def taylor_ver_vaf(self,alpha_in, adjoint_flag=0):
        self.alpha = alpha_in
        self.H = self.H_init.copy(deepcopy=True)
        self.H_s = self.H_init.copy(deepcopy=True)
        self.H_np = self.H_init.copy(deepcopy=True)
        self.timestep(save=0, adjoint_flag=adjoint_flag)
        self.set_J_vaf()
        return assemble(self.J_vaf)


class ddJ_wrapper(object):
    def __init__(self, ddJ_action, cntrl):
        self.ddJ_action = ddJ_action
        self.ddJ_F = Function(cntrl.function_space())

    def apply(self,x):
        self.ddJ_F.vector().set_local(x.getArray())
        self.ddJ_F.vector().apply('insert')
        return self.ddJ_action(self.ddJ_F).vector().get_local()


class MomentumSolver(EquationSolver):

    def __init__(self, *args, **kwargs):
        self.picard_params = kwargs.pop("picard_params", None)
        self.J_p = kwargs.pop("J_p", None)
        super(MomentumSolver, self).__init__(*args, **kwargs)

    def forward_solve(self, x, deps):
        #replace_map = dict(zip(self._EquationSolver__deps, deps))
        replace_map = dict(zip(self.dependencies(), deps))

        if not self._EquationSolver__initial_guess is None:
          x.assign(replace_map[self._EquationSolver__initial_guess])
        replace_map[self.x()] = x

        lhs = replace(self._EquationSolver__lhs, replace_map)
        rhs = 0 if self._EquationSolver__rhs == 0 else replace(self._EquationSolver__rhs, replace_map)
        J = replace(self._EquationSolver__J, replace_map)

        solve(lhs == rhs, x, self._EquationSolver__bcs, J = replace(self.J_p, replace_map), form_compiler_parameters = self._EquationSolver__form_compiler_parameters, solver_parameters = self.picard_params)
        solve(lhs == rhs, x, self._EquationSolver__bcs, J = J, form_compiler_parameters = self._EquationSolver__form_compiler_parameters, solver_parameters = self._EquationSolver__solver_parameters)

        return
