#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import Functional, InnerProductSolver, clear_caches, \
    function_axpy, function_copy, function_is_cached, \
    function_is_checkpointed, function_is_static, function_name, \
    function_new, set_manager
from tlm_adjoint import manager as _manager
from IPython import embed

from collections.abc import Sequence

__all__ = \
    [
        "GaussNewton"
    ]


class GaussNewton:
    def __init__(self, forward, R_inv_action, manager=None):
        if manager is None:
            manager = _manager().new()

        self._forward = forward
        #self._B_inv_action = B_inv_action
        self._R_inv_action = R_inv_action
        self._manager = manager

    def action(self, M, dM):
        if not isinstance(M, Sequence):
            ddJ, = self.action((M,), (dM,))
            return ddJ

        old_manager = _manager()
        manager = self._manager
        set_manager(manager)
        manager.reset()
        manager.stop()
        clear_caches()

        M = [function_copy(m, name=function_name(m),
                           static=function_is_static(m),
                           cache=function_is_cached(m),
                           checkpoint=function_is_checkpointed(m)) for m in M]

        # Possible optimization: We could add multiple TLMs to compute multiple
        # actions with a single forward run
        manager.add_tlm(M, dM)
        # Possible optimization: We annotate all the TLM equations, but are
        # later only going to differentiate back through the forward -- so this
        # uses unnecessary storage
        manager.start()
        X = self._forward(*M)
        if not isinstance(X, Sequence):
            X = (X,)
        manager.stop()

        # J dM
        tau_X = [manager.tlm(M, dM, x) for x in X]
        # R^{-1} J dM
        R_inv_tau_X = self._R_inv_action(*tau_X)
        if not isinstance(R_inv_tau_X, Sequence):
            R_inv_tau_X = (R_inv_tau_X,)

        # This construction defines the adjoint right-hand-side appropriately,
        # to compute a J^T action
        manager.start()
        J = Functional(name="J")
        for x, R_inv_tau_x in zip(X, R_inv_tau_X):
            J_term = function_new(J.fn())
            InnerProductSolver(function_copy(R_inv_tau_x), x, J_term).solve()
            J.addto(J_term)
        manager.stop()

        # Likelihood term: J^T R^{-1} J dM
        H_GN_action = manager.compute_gradient(J, M)

        # Prior term
        #B_inv_dM = self._B_inv_action(*dM)
        #if not isinstance(B_inv_dM, Sequence):
        #    B_inv_dM = (B_inv_dM,)
        #for i, B_inv_dm in enumerate(B_inv_dM):
        #    function_axpy(H_GN_action[i], 1.0, B_inv_dm)

        set_manager(old_manager)
        return H_GN_action
