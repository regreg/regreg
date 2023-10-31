from __future__ import print_function, division, absolute_import

from warnings import warn
from copy import copy
import gc

import numpy as np
import numpy.linalg as npl

from scipy.stats import rankdata

from . import subsample_columns
from ..affine import power_L, normalize, astransform
from ..smooth import glm, affine_smooth, sum as smooth_sum
from ..smooth.quadratic import quadratic_loss
from ..problems.simple import simple_problem
from ..identity_quadratic import identity_quadratic as iq
from ..atoms.sparse_group_lasso import (sparse_group_lasso,
                                        _gauge_function_dual_strong,
                                        _inside_set_strong)
from .group_lasso import group_lasso_path, default_lagrange_sequence

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
                 l1_alpha=0.95, # mix between l1 and l2 penalty
                 ):

        self.saturated_loss = saturated_loss
        self.X = astransform(X)
        self.l1_alpha = l1_alpha # used only in constructor and 
                                 # in subsample

        # the penalty parameters

        self.alpha = alpha
        self.penalty = sparse_group_lasso(groups, lasso_weights, weights=weights, lagrange=1)
        l2_weight = 1 - l1_alpha
        for g in self.penalty.weights.keys():
            self.penalty.set_weight(g, l2_weight * self.penalty.weights[g])
        self.penalty.lasso_weights = self.penalty.lasso_weights * l1_alpha
        self.group_shape = (len(np.unique(self.penalty.groups)),)
        self.shape = self.penalty.shape

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
             _) = self.solve_subproblem(self.solution,
                                        unpenalized_groups,
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
                unpen_group = (self.penalty.weights[g] == 0 +
                               np.any(self.penalty.lasso_weights[self.penalty.groups == g]
                                      == 0))
                self._unpen_group_idx[i] = unpen_group
                if unpen_group:
                    self._unpen_groups.append(g)
        return self._unpen_groups, self._unpen_group_idx

    # methods potentially overwritten in subclasses for I/O considerations

    def subsample(self,
                  case_idx):
        '''

        Create a new path, by subsampling
        cases of `self.saturated_loss`.

        Case weights are computed
        with `self.saturated_loss.subsample`.

        Parameters
        ----------

        case_idx : index
            An index-like object used 
            to specify which cases to include
            in the subsample.

        Returns
        -------

        subsample_path : path object
            A path object with a modified smooth part
            reflecting the subsampling.

        '''
        subsample_loss = self.saturated_loss.subsample(case_idx)
        return self.__class__(subsample_loss,
                              self.X,
                              self.penalty.groups,
                              self.penalty.lasso_weights,
                              weights=self.penalty.weights,
                              elastic_net_param=self.elastic_net_param,
                              alpha=self.alpha,
                              l1_alpha=self.l1_alpha)

    def check_KKT(self,
                  grad_solution,
                  solution,
                  lagrange,
                  penalty=None):

        '''

        Check KKT conditions over
        the groups in the path.
        Returns boolean indicating
        which groups are failing the KKT conditions
        (these could be `active` groups or
        `inactive` groups).

        Parameters
        ----------

        grad_solution : ndarray
             Candidate for gradient of smooth loss at 
             Lagrange value `lagrange`.

        solution : ndarray
             Candidate for solution to problem 
             Lagrange value `lagrange`.

        lagrange : float
             Lagrange value for penalty

        penalty : object (optional)
             A sparse group LASSO penalty. If None, defaults
             to `self.penalty`.

        '''
        if penalty is None:
            penalty = self.penalty

        active, inactive_ranks = _check_KKT(penalty,
                                            grad_solution, 
                                            solution, 
                                            lagrange)
        return active > 0, inactive_ranks


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
                                             
    def solve_subproblem(self, 
                         solution,
                         candidate_groups, 
                         lagrange_new, 
                         **solve_args):
    
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
            sub_elastic_net = self.enet_loss(lagrange_new,
                                             candidate_groups)

            sub_loss = smooth_sum([sub_loss, sub_elastic_net])

        sub_problem = simple_problem(sub_loss, sub_penalty)
        sub_problem.coefs[:] = solution[candidate_bool] # warm start
        sub_soln = sub_problem.solve(**solve_args)
        sub_grad = sub_loss.smooth_objective(sub_soln, mode='grad') 
        sub_linear_pred = sub_X.dot(sub_soln)
        return sub_problem.final_step, sub_grad, sub_soln, sub_linear_pred, candidate_bool

    def restricted_penalty(self, subset):
        if subset is not None:
            groups = self.penalty.groups[subset]
            lasso_weights = self.penalty.lasso_weights[subset]
        else:
            groups = self.penalty.groups
            lasso_weights = self.penalty.lasso_weights
        return sparse_group_lasso(groups,
                                  lasso_weights,
                                  weights=self.penalty.weights,
                                  lagrange=1)#
# Some common loss factories

def logistic(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return sparse_group_lasso_path(glm.logistic_loglike(Y.shape, Y), X, *args, **keyword_args)

def probit(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return sparse_group_lasso_path(glm.probit_loglike(Y.shape, Y), X, *args, **keyword_args)

def cloglog(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return sparse_group_lasso_path(glm.cloglog_loglike(Y.shape, Y), X, *args, **keyword_args)

def gaussian(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return sparse_group_lasso_path(glm.gaussian_loglike(Y.shape, Y), X, *args, **keyword_args)

def cox(X, T, S, *args, **keyword_args):
    T, S = np.asarray(T), np.asarray(S)
    return sparse_group_lasso_path(glm.cox_loglike(T.shape, T, S), X, *args, **keyword_args)

def poisson(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return sparse_group_lasso_path(glm.poisson_loglike(Y.shape, Y), X, *args, **keyword_args)

def huber(X, Y, smoothing_parameter, *args, **keyword_args):
    Y = np.asarray(Y)
    return sparse_group_lasso_path(glm.huber_loss(Y.shape, Y, smoothing_parameter), X, *args, **keyword_args)

def huber_svm(X, Y, smoothing_parameter, *args, **keyword_args):
    Y = np.asarray(Y)
    return sparse_group_lasso_path(glm.huber_svm(Y.shape, Y, smoothing_parameter), X, *args, **keyword_args)

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
                gradient,
                slope_estimate=1):

    """
    Return a Boolean indicator for each group
    indicating whether in the strong set or not.
    """

    thresh = (slope_estimate + 1) * lagrange_new - slope_estimate * lagrange_cur
    thresh = (slope_estimate + 1) * lagrange_new - slope_estimate * lagrange_cur
    test = np.zeros(len(penalty._sorted_groupids), np.bool)
    for i, g in enumerate(penalty._sorted_groupids):
        group = penalty.groups == g
        test[i] = _inside_set_strong(gradient[group],
                                     thresh,
                                     penalty.lasso_weights[group],
                                     penalty.weights[g]) == False
    return test

# def _restricted_elastic_net(elastic_net_params, 
#                             penalized,
#                             groups,
#                             lagrange, 
#                             alpha,
#                             candidate_groups):

#     candidate_bool = _candidate_bool(groups, candidate_groups)

#     new_params = elastic_net_params * (1 - alpha)
#     new_params[penalized] *= lagrange 
#     new_params = new_params[candidate_bool]
#     return quadratic_loss(new_params.shape,
#                           new_params,
#                           Qdiag=True)

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

    active_results = np.zeros(len(sglasso._sorted_groupids), int)
    inactive_results = np.zeros(len(sglasso._sorted_groupids), float)

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
                active_results[i] = ACTIVE_NORM
            nonz = soln_g != 0

            # nonzero coordinates need the right sign and size
            if (np.linalg.norm((l1subgrad_g - l1weight_g * np.sign(soln_g) * lagrange)[nonz]) > 
                tol * max(1, np.linalg.norm(soln_g))):
                active_results[i] = ACTIVE_L1

            # l2 subgrad should be parallel to soln_g
            if np.linalg.norm(l2subgrad_g / np.linalg.norm(l2subgrad_g) - 
                              soln_g / np.linalg.norm(soln_g)) > tol:
                active_results[i] = ACTIVE_L2
        elif l1weight_g.sum() + sglasso.weights[g] > 0: # inactive penalized
            inactive_results[i] = _gauge_function_dual_strong(subgrad_g, l1weight_g,
                                                              l2weight_g)[0] / lagrange
        else:   # unpenalized
            active_results[i] = (np.linalg.norm(subgrad_g) > tol * sqlasso.l1weight_g.mean()) * UNPENALIZED

    inactive_ranks = inactive_results.shape[0] - rankdata(inactive_results)
    inactive_ranks[inactive_results <= 1 + tol] = -1

    return active_results, inactive_ranks

