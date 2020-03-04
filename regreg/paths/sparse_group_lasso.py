from __future__ import print_function, division, absolute_import

from warnings import warn
from copy import copy
import gc

import numpy as np
import numpy.linalg as npl

import scipy.sparse

from . import subsample_columns
from ..affine import power_L, normalize, astransform
from ..smooth import glm, affine_smooth, sum as smooth_sum
from ..smooth.quadratic import quadratic_loss
from ..problems.simple import simple_problem
from ..identity_quadratic import identity_quadratic as iq
from ..atoms.sparse_group_lasso import (sparse_group_lasso,
                                        _gauge_function_dual_strong)
from .group_lasso import group_lasso_path

class sparse_group_lasso_path(group_lasso_path):

    BIG = 1e12 # lagrange parameter for finding null solution

    def __init__(self, 
                 saturated_loss,
                 X, 
                 groups,
                 lasso_weights,
                 weights={},
                 elastic_net_param=None,
                 alpha=1.,  # elastic net mixing -- 1 is LASSO
                 l1_weight=0.95, # mix between l1 and l2 penalty
                 lagrange_proportion=0.05,
                 nstep=100,
                 elastic_net_penalized=None):

        self.saturated_loss = saturated_loss
        self.X = astransform(X)

        # the penalty parameters

        self.alpha = alpha
        self.penalty = sparse_group_lasso(groups, lasso_weights, weights=weights, lagrange=1)
        l2_weight = 1 - l1_weight
        for g in self.penalty.weights.keys():
            self.penalty.set_weight(g, l2_weight * self.penalty.weights[g])
        self.penalty.lasso_weights *= l1_weight
        self.group_shape = (len(np.unique(self.penalty.groups)),)
        self.shape = self.penalty.shape
        self.nstep = nstep

        # elastic net part
        if elastic_net_param is None:
            elastic_net_param = np.ones(self.shape)
        self.elastic_net_param = elastic_net_param

        # find lagrange_max

        unpenalized_groups, unpenalized_idx = self.unpenalized

        self.solution = np.zeros(self.penalty.shape)

        self.ever_active_groups = self.updated_ever_active(unpenalized_idx)

        self._unpenalized_vars = _candidate_bool(self.penalty.groups, 
                                                 unpenalized_groups)
        self._penalized_vars = np.zeros(self.shape, np.bool)
        self._penalized_vars[self._unpenalized_vars] = 0

        if np.any(unpenalized_idx):
            (self.final_step, 
             null_grad, 
             null_soln,
             null_linpred,
             _) = self.solve_subproblem(unpenalized_groups,
                                        self.BIG,
                                        tol=1.e-8)
            self.linear_predictor = null_linpred
            self.solution[self._unpenalized_vars] = null_soln
        else:
            self.linear_predictor = np.zeros(self.saturated_loss.shape)

        if np.any(self.elastic_net_param[self._unpenalized_vars]):
            warn('convention is that unpenalized parameters with have no Lagrange parameter in front '
                 'of their ridge term so that lambda_max is easily computed')

        self.grad_solution = (self.full_gradient(self.saturated_loss, 
                                                 self.linear_predictor) + self.enet_grad(self.solution, 
                                                                                         self._penalized_vars,
                                                                                         1))
        self.lagrange_max = self.get_lagrange_max(self.grad_solution) # penalty specific
        self.lagrange_sequence = self.lagrange_max * np.exp(np.linspace(np.log(lagrange_proportion), 
                                                                        0, 
                                                                        nstep))[::-1]

    # methods potentially overwritten in subclasses for I/O considerations

    def check_KKT(self,
                  grad_solution,
                  solution,
                  lagrange_new,
                  penalty=None):

        if penalty is None:
            penalty = self.penalty

        results = _check_KKT(penalty,
                             grad_solution, 
                             solution, 
                             lagrange_new)
        return results > 0

    def strong_set(self,
                   lagrange_cur,
                   lagrange_new,
                   grad_solution):

        _strong_bool = _strong_set(self.penalty,
                                   self._unpen_group_idx,
                                   lagrange_cur,
                                   lagrange_new,
                                   grad_solution)
        _strong = np.array([self.penalty._sorted_groupids[i] for i in np.nonzero(_strong_bool)[0]])
        return (_strong, 
                _strong_bool,
                np.nonzero(_candidate_bool(self.penalty.groups, _strong))[0])
                                             
    def solve_subproblem(self, candidate_groups, lagrange_new, **solve_args):
    
        # solve a problem with a candidate set

        sub_loss, sub_penalty, sub_X, candidate_bool = _restricted_problem(self.X, 
                                                                           self.saturated_loss, 
                                                                           self.alpha * lagrange_new, 
                                                                           self.penalty.groups,
                                                                           self.penalty.lasso_weights,
                                                                           self.penalty.weights,
                                                                           candidate_groups,
                                                                           self.subsample_columns)
        if self.alpha < 1:
            sub_elastic_net = _restricted_elastic_net(self.elastic_net_param, 
                                                      self._penalized_vars,
                                                      self.penalty.groups,
                                                      lagrange_new,
                                                      self.alpha,
                                                      candidate_groups)

            sub_loss = smooth_sum([sub_loss, sub_elastic_net])

        sub_problem = simple_problem(sub_loss, sub_penalty)
        sub_problem.coefs[:] = self.solution[candidate_bool] # warm start
        sub_soln = sub_problem.solve(**solve_args)
        sub_grad = sub_loss.smooth_objective(sub_soln, mode='grad') 
        sub_linear_pred = sub_X.dot(sub_soln)
        return sub_problem.final_step, sub_grad, sub_soln, sub_linear_pred, candidate_bool

    def restricted_penalty(self, subset):
        return sparse_group_lasso(self.penalty.groups[subset],
                                  self.penalty.lasso_weights[subset],
                                  weights=self.penalty.weights,
                                  lagrange=1)
# private functions

def _candidate_bool(groups, candidate_groups):

    candidate_bool = np.zeros(groups.shape, np.bool)
    for g in candidate_groups:
        group = groups == g
        candidate_bool += group

    return candidate_bool

def _strong_set(penalty,
                unpenalized_idx, 
                lagrange_cur,
                lagrange_new,
                gradient):

    dual = penalty.conjugate
    prox_grad = penalty.lagrange_prox(gradient, lagrange=2*lagrange_new-lagrange_cur)
    terms = penalty.terms(prox_grad)
    value = np.asarray(terms) > 0
    value[unpenalized_idx] = True
    return value

def _restricted_elastic_net(elastic_net_params, 
                            penalized,
                            groups,
                            lagrange, 
                            alpha,
                            candidate_groups):

    candidate_bool = _candidate_bool(groups, candidate_groups)

    new_params = elastic_net_params * (1 - alpha)
    new_params[penalized] *= lagrange 
    new_params = new_params[candidate_bool]
    return quadratic_loss(new_params.shape,
                          new_params,
                          Qdiag=True)

def _restricted_problem(X, 
                        saturated_loss, 
                        alpha,
                        groups,
                        lasso_weights,
                        group_weights, 
                        candidate_groups,
                        subsample_columns):

    candidate_bool = _candidate_bool(groups, candidate_groups)
    X_candidate = subsample_columns(X, np.nonzero(candidate_bool)[0])
    restricted_penalty = sparse_group_lasso(groups[candidate_bool],
                                            lasso_weights[candidate_bool],
                                            weights=copy(group_weights), # should be OK to reuse weights dict
                                            lagrange=alpha)

    restricted_loss = affine_smooth(saturated_loss, X_candidate)
    return restricted_loss, restricted_penalty, X_candidate, candidate_bool

# for paths

def _check_KKT(sglasso, 
               grad, 
               solution, 
               lagrange, 
               tol=1.e-2):

    """
    Check whether (grad, solution) satisfy
    KKT conditions at a given tolerance.

    Assumes glasso is group_lasso in lagrange form
    so that glasso.lagrange is not None
    """

    terms = sglasso.terms(solution)
    norm_soln = np.linalg.norm(solution)

    ACTIVE_L1 = 10
    ACTIVE_NORM = 11
    ACTIVE_L2 = 12
    INACTIVE = 2
    UNPENALIZED = 3

    results = np.zeros(len(sglasso._sorted_groupids), np.int)
    for i, g in enumerate(sglasso._sorted_groupids):
        group = sglasso.groups == g
        subgrad_g = -grad[group]
        l1weight_g = sglasso.lasso_weights[group]
        l2weight_g = sglasso.weights[g]
        soln_g = solution[group]

        if terms[i] > 1.e-6 * norm_soln: # active group
            val_g, l1subgrad_g, l2subgrad_g = _gauge_function_dual_strong(subgrad_g, 
                                                                          l1weight_g,
                                                                          l2weight_g)
            if val_g < lagrange * (1 - tol):
                results[i] = ACTIVE_NORM
            nonz = soln_g != 0

            # nonzero coordinates need the right sign and size
            if (np.linalg.norm((l1subgrad_g - l1weight_g * np.sign(soln_g) * lagrange)[nonz]) > 
                tol * max(1, np.linalg.norm(soln_g))):
                results[i] = ACTIVE_L1

            # l2 subgrad should be parallel to soln_g
            if np.linalg.norm(l2subgrad_g / np.linalg.norm(l2subgrad_g) - 
                              soln_g / np.linalg.norm(soln_g)) > tol:
                results[i] = ACTIVE_L2
        elif l1weight_g.sum() + sglasso.weights[g] > 0: # inactive penalized
            results[i] = (_gauge_function_dual_strong(subgrad_g, l1weight_g,
                                                      l2weight_g)[0] >= lagrange * (1 + tol)) * INACTIVE
        else:   # unpenalized
            results[i] = (np.linalg.norm(subgrad_g) > tol * sqlasso.l1weight_g.mean()) * UNPENALIZED
    return results
