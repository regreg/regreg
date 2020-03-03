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
from ..atoms.group_lasso import group_lasso
from .lasso import lasso_path

class group_lasso_path(lasso_path):

    BIG = 1e12 # lagrange parameter for finding null solution

    def __init__(self, 
                 saturated_loss,
                 X, 
                 groups,
                 weights={},
                 elastic_net_param=None,
                 alpha=1.,  # elastic net mixing -- 1 is LASSO
                 lagrange_proportion=0.05,
                 nstep=100,
                 elastic_net_penalized=None):

        self.saturated_loss = saturated_loss
        self.X = astransform(X)

        # the penalty parameters

        self.alpha = alpha
        self.penalty = group_lasso(groups, weights=weights, lagrange=1)
        self.group_shape = len(np.unique(self.penalty.groups))
        self.shape = self.penalty.shape
        self.nstep = nstep

        # elastic net part
        if elastic_net_param is None:
            elastic_net_param = np.ones(self.shape)
        self.elastic_net_param = elastic_net_param

        # find lagrange_max

        unpenalized_groups, unpenalized_idx = self.unpenalized
        print('unpen', unpenalized_groups)
        self.solution = np.zeros(self.penalty.shape)

        self.ever_active_groups = self.updated_ever_active(unpenalized_idx)

        self._unpenalized_vars = _candidate_bool(self.penalty.groups, 
                                                 unpenalized_groups)
        self._penalized_vars = np.ones(self.shape, np.bool)
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

    def main2(self, inner_tol=1.e-5, verbose=False):

        _lipschitz = power_L(self.X)

        # take a guess at the inverse step size
        self.final_step = 1000. / _lipschitz 
        lseq = self.lagrange_sequence # shorthand

        # gradient of restricted elastic net at lambda_max

        solution = self.solution
        grad_solution = self.grad_solution
        linear_predictor = self.linear_predictor
        ever_active = self.updated_ever_active([])

        obj_solution = self.saturated_loss.smooth_objective(self.linear_predictor, 'func')
        solutions = []
        objective = [obj_solution]

        all_failing = np.zeros(self.group_shape, np.bool)
        subproblem_set = self.updated_ever_active([])

        for lagrange_new, lagrange_cur in zip(lseq[1:], lseq[:-1]):
            tol = inner_tol
            num_tries = 0
            debug = False
            coef_stop = True

            while True:

                subproblem_set = sorted(set(subproblem_set + self.updated_ever_active(all_failing)))

                (self.final_step, 
                 subproblem_grad, 
                 subproblem_soln,
                 subproblem_linpred,
                 subproblem_vars) = self.solve_subproblem(subproblem_set,
                                                          lagrange_new,
                                                          tol=tol,
                                                          start_step=self.final_step,
                                                          debug=debug and verbose,
                                                          coef_stop=coef_stop)

                saturated_grad = self.saturated_loss.smooth_objective(subproblem_linpred, 'grad')
                # as subproblem always contains ever active, 
                # rest of solution should be 0
                solution[subproblem_vars] = subproblem_soln

                # strong rules step
                # a set of group ids

                strong, strong_idx = self.strong_set(lagrange_cur * self.alpha, 
                                                     lagrange_new * self.alpha, 
                                                     grad_solution)
                strong_vars = np.nonzero(_candidate_bool(self.penalty.groups, strong))[0]
                strong_enet_grad = self.enet_grad(solution,
                                                  self._penalized_vars,
                                                  lagrange_new,
                                                  subset=strong_vars)
                strong_soln = solution[strong_vars]
                X_strong = self.subsample_columns(self.X, 
                                                  strong_vars)
                strong_grad = (X_strong.T.dot(saturated_grad) +
                               strong_enet_grad)
                strong_penalty = self.restricted_penalty(strong_vars)
                strong_failing = self.check_KKT(strong_grad, 
                                                strong_soln, 
                                                self.alpha * lagrange_new, 
                                                penalty=strong_penalty)

                if np.any(strong_failing):
                    delta = np.zeros(self.group_shape, np.bool)
                    delta[strong_idx] = strong_failing
                    all_failing += delta 
                else:
                    enet_grad = self.enet_grad(solution, 
                                               self._penalized_vars,
                                               lagrange_new)
                    grad_solution[:] = (self.full_gradient(self.saturated_loss, 
                                                           subproblem_linpred) + 
                                        enet_grad)
                    all_failing = self.check_KKT(grad_solution, 
                                                 solution, 
                                                 self.alpha * lagrange_new)

                    if not all_failing.sum():
                        self.ever_active = self.updated_ever_active(self.active_set(solution))
                        linear_predictor[:] = subproblem_linpred
                        break
                    else:
                        if verbose:
                            print('failing:', np.nonzero(all_failing)[0])
                num_tries += 1

                tol /= 2.
                if num_tries % 5 == 0:

                    solution[subproblem_vars] = subproblem_soln
                    solution[~subproblem_vars] = 0

                    enet_grad = self.enet_grad(solution, 
                                               self._penalized_vars,
                                               lagrange_new)
                    grad_solution[:] = (self.full_gradient(self.saturated_loss, 
                                                           subproblem_linpred) + 
                                        enet_grad)
                    debug = True
                    tol = inner_tol
                    if num_tries >= 20:
                        warn('convergence not achieved for lagrange=%0.4e' % lagrange_new)
                        break

                subproblem_set = sorted(set(subproblem_set + self.updated_ever_active(all_failing)))

            solutions.append(solution.copy())
            objective.append(self.saturated_loss.smooth_objective(self.linear_predictor, mode='func'))
            gc.collect()

            if verbose:
                print(lagrange_cur / self.lagrange_max,
                      lagrange_new,
                      (solution != 0).sum(),
                      1. - objective[-1] / objective[0],
                      list(self.lagrange_sequence).index(lagrange_new),
                      np.fabs(rescaled_solution).sum())

        objective = np.array(objective)
        output = {'devratio': 1 - objective / objective.max(),
                  'lagrange': self.lagrange_sequence,
                  'beta':np.array(solutions)}

        return output

    # method potentially overwritten in subclasses for penalty considerations

    def get_lagrange_max(self,
                         grad_solution):
        dual = self.penalty.conjugate
        return dual.seminorm(grad_solution, lagrange=1)

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

    def enet_grad(self,
                  solution,
                  penalized, # boolean
                  lagrange_new,
                  subset=None):

        weights = self.penalty.weights
        elastic_net_param = self.elastic_net_param
        if subset is not None:
            solution = solution[subset]
            penalized = penalized[subset]
            elastic_net_param = elastic_net_param[subset]
        G = (1 - self.alpha) * solution * elastic_net_param
        G[penalized] *= lagrange_new
        return G

    def updated_ever_active(self,
                            index_obj):
        if not hasattr(self, '_ever_active'):
            self._ever_active = np.zeros(self.group_shape, np.bool)
            self._sorted_groupids = np.array(self.penalty._sorted_groupids)
        _ever_active = self._ever_active.copy()
        _ever_active[index_obj] = True
        return list(self._sorted_groupids[_ever_active])

    @property
    def unpenalized(self):
        """
        Unpenalized groups and variables.

        Returns
        -------

        groups : sequence
            Groups with weights equal to 0.

        variables : ndarray
            Boolean indicator that is True if no penalty on that variable.

        """
        if not hasattr(self, "_unpen_groups"):
            self._unpen_groups = []
            self._unpen_group_idx = np.zeros(self.group_shape, np.bool)
            for i, g in enumerate(self.penalty._sorted_groupids):
                unpen_group = self.penalty.weights[g] == 0
                self._unpen_group_idx[i] = unpen_group
                if unpen_group:
                    self._unpen_groups.append(g)
        return self._unpen_groups, self._unpen_group_idx

    def active_set(self, solution):
        return self.penalty.terms(solution) != 0

    def restricted_penalty(self, subset):
        return group_lasso(self.penalty.groups[subset],
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
                lagrange_cur,
                lagrange_new,
                gradient):

    dual = penalty.conjugate
    return np.asarray(dual.terms(gradient, check_feasibility=True)) > 2 * lagrange_new - lagrange_cur

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
                        group_weights, 
                        candidate_groups,
                        subsample_columns):

    candidate_bool = _candidate_bool(groups, candidate_groups)
    X_candidate = subsample_columns(X, np.nonzero(candidate_bool)[0])
    restricted_penalty = group_lasso(groups[candidate_bool],
                                     weights=copy(group_weights), # should be OK to reuse weights dict
                                     lagrange=alpha)

    restricted_loss = affine_smooth(saturated_loss, X_candidate)
    return restricted_loss, restricted_penalty, X_candidate, candidate_bool

def _check_KKT(penalty,
               grad, 
               solution, 
               lagrange, 
               tol=1.e-2):

    """
    Check whether (grad, solution) satisfy
    KKT conditions at a given tolerance.
    """
    dual = penalty.conjugate
    terms = dual.terms(solution)

    ACTIVE = 1
    INACTIVE = 2
    UNPENALIZED = 3

    results = np.zeros(len(penalty._sorted_groupids), np.int)
    for i, g in enumerate(penalty._sorted_groupids):
        group = penalty.groups == g
        w = penalty.weights[g]
        active = (terms[i] != 0) * (w > 0)
        unpenalized = w == 0
        if active:
            results[i] = (np.linalg.norm(solution[group] * lagrange * w / np.linalg.norm(solution[group]) + grad[group]) > tol) * ACTIVE
        elif unpenalized:
            results[i] = (np.linalg.norm(grad[group]) > tol * np.mean(penalty._weight_array)) * UNPENALIZED
        else:
            results[i] = (np.linalg.norm(grad[group]) / (lagrange * w) > 1 + tol) * INACTIVE
    return results
