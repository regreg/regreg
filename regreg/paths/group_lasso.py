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
from ..atoms.group_lasso import group_lasso
from . import grouped_path, default_lagrange_sequence

class group_lasso_path(grouped_path):

    BIG = 1e12 # lagrange parameter for finding null solution

    def __init__(self, 
                 saturated_loss,
                 X, 
                 groups,
                 weights={},
                 elastic_net_param=None,
                 alpha=1.  # elastic net mixing -- 1 is LASSO
                 ):

        self.saturated_loss = saturated_loss
        self.X = astransform(X)

        # the penalty parameters

        self.alpha = alpha
        self.penalty = group_lasso(groups, weights=weights, lagrange=1)
        self.group_shape = (len(np.unique(self.penalty.groups)),)
        self.shape = self.penalty.shape

        # elastic net part
        if elastic_net_param is None:
            elastic_net_param = np.ones(self.shape)
        self.elastic_net_param = elastic_net_param

        unpenalized_groups, unpenalized_idx = self.unpenalized
        self.solution = np.zeros(self.penalty.shape)

        self._unpenalized_vars = _candidate_bool(self.penalty.groups, 
                                                 unpenalized_groups)
        self._penalized_vars = np.ones(self.shape, np.bool)
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

    # method potentially overwritten in subclasses for penalty considerations

    def enet_loss(self,
                  lagrange,
                  candidate_groups=None):

        return _restricted_elastic_net(self.elastic_net_param, 
                                       self._penalized_vars,
                                       self.penalty.groups,
                                       lagrange,
                                       self.alpha,
                                       candidate_groups)

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
                              weights=self.penalty.weights,
                              elastic_net_param=self.elastic_net_param,
                              alpha=self.alpha)

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
             A group LASSO penalty. If None, defaults
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

    def enet_grad(self,
                  solution,
                  penalized, # boolean
                  lagrange_new,
                  subset=None):

        elastic_net_param = self.elastic_net_param
        if subset is not None:
            solution = solution[subset]
            penalized = penalized[subset]
            elastic_net_param = elastic_net_param[subset]
        G = (1 - self.alpha) * solution * elastic_net_param
        G[penalized] *= lagrange_new
        return G

    def updated_ever_active(self,
                            index_obj,
                            group_ids=True):
        if not hasattr(self, '_ever_active'):
            self._ever_active = np.zeros(self.group_shape, np.bool)
            self._sorted_groupids = np.array(self.penalty._sorted_groupids)
        _ever_active = self._ever_active.copy()
        _ever_active[index_obj] = True
        if group_ids:
            return list(self._sorted_groupids[_ever_active])
        else:
            return list(np.nonzero(_ever_active)[0])

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

    def restricted_penalty(self, var_subset):
        if var_subset is not None:
            groups = self.penalty.groups[var_subset]
        else:
            groups = self.penalty.groups
        return group_lasso(groups,
                           weights=self.penalty.weights,
                           lagrange=1)

# Some common loss factories

def logistic(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return group_lasso_path(glm.logistic_loglike(Y.shape, Y), X, *args, **keyword_args)

def probit(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return group_lasso_path(glm.probit_loglike(Y.shape, Y), X, *args, **keyword_args)

def cloglog(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return group_lasso_path(glm.cloglog_loglike(Y.shape, Y), X, *args, **keyword_args)

def gaussian(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return group_lasso_path(glm.gaussian_loglike(Y.shape, Y), X, *args, **keyword_args)

def cox(X, T, S, *args, **keyword_args):
    T, S = np.asarray(T), np.asarray(S)
    return group_lasso_path(glm.cox_loglike(T.shape, T, S), X, *args, **keyword_args)

def poisson(X, Y, *args, **keyword_args):
    Y = np.asarray(Y)
    return group_lasso_path(glm.poisson_loglike(Y.shape, Y), X, *args, **keyword_args)

def huber(X, Y, smoothing_parameter, *args, **keyword_args):
    Y = np.asarray(Y)
    return group_lasso_path(glm.huber_loss(Y.shape, Y, smoothing_parameter), X, *args, **keyword_args)

def huber_svm(X, Y, smoothing_parameter, *args, **keyword_args):
    Y = np.asarray(Y)
    return group_lasso_path(glm.huber_svm(Y.shape, Y, smoothing_parameter), X, *args, **keyword_args)

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
                gradient,
                slope_estimate=1):
    """
    Return a Boolean indicator for each group
    indicating whether in the strong set or not.
    """

    thresh = (slope_estimate + 1) * lagrange_new - slope_estimate * lagrange_cur
    dual = penalty.conjugate
    return np.asarray(dual.terms(gradient, check_feasibility=True)) > thresh

def _restricted_elastic_net(elastic_net_params, 
                            penalized,
                            groups,
                            lagrange, 
                            alpha,
                            candidate_groups):

    new_params = elastic_net_params * (1 - alpha)
    new_params[penalized] *= lagrange 
    if candidate_groups is not None:
        candidate_bool = _candidate_bool(groups, candidate_groups)
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

    active_results = np.zeros(len(penalty._sorted_groupids), int)
    inactive_results = np.zeros(len(penalty._sorted_groupids), float)
    for i, g in enumerate(penalty._sorted_groupids):
        group = penalty.groups == g
        w = penalty.weights[g]
        active = (terms[i] != 0) * (w > 0)
        unpenalized = w == 0
        if active:
            active_results[i] = (np.linalg.norm(solution[group] * lagrange * w / np.linalg.norm(solution[group]) + grad[group]) > tol) * ACTIVE
        elif unpenalized:
            active_results[i] = (np.linalg.norm(grad[group]) > tol * np.mean(penalty._weight_array)) * UNPENALIZED
        else:
            inactive_results[i] = np.linalg.norm(grad[group]) / (lagrange * w) 
    # rank inactive groups

    inactive_ranks = inactive_results.shape[0]  - rankdata(inactive_results)
    inactive_ranks[inactive_results <= 1 + tol] = -1

    return active_results, inactive_ranks
