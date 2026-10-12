"""
Regularization paths for the (weighted) LASSO and elastic net.
"""

from __future__ import print_function, division, absolute_import

from warnings import warn
import numpy as np
import numpy.linalg as npl

from scipy.stats import rankdata

from . import subsample_columns, grouped_path, default_lagrange_sequence
from ..affine import astransform, power_L
from ..smooth import glm, affine_smooth, sum as smooth_sum
from ..smooth.quadratic import quadratic_loss
from ..problems.simple import simple_problem
from ..identity_quadratic import identity_quadratic as iq
from ..atoms.weighted_atoms import l1norm as weighted_l1norm

class lasso_path(grouped_path):

    r"""
    Regularization path for the weighted LASSO / elastic net.

    The problem at Lagrange parameter :math:`\lambda` is

    .. math::

        \ell(X\beta) + \lambda \alpha \sum_j w_j |\beta_j| +
        \frac{1 - \alpha}{2} \sum_j \lambda_j c_j \beta_j^2

    where :math:`\ell` is `saturated_loss`, :math:`w` is `lasso_weights`,
    :math:`c` is `elastic_net_param`, and :math:`\lambda_j=\lambda` for
    penalized variables, :math:`\lambda_j=1` for unpenalized ones.

    Parameters
    ----------
    saturated_loss : `regreg.smooth.glm.glm`
        Loss as a function of the linear predictor.
    X : ndarray or `regreg.affine.affine_transform`
        Design matrix.
    lasso_weights : ndarray
        Weights for the L1 penalty; variables with weight 0 are unpenalized.
    elastic_net_param : ndarray, optional
        Ridge weights for each variable. Defaults to all ones.
    alpha : float, optional
        Elastic net mixing parameter; 1 is the LASSO.
    """

    BIG = 1e12 # lagrange parameter for finding null solution

    def __init__(self, 
                 saturated_loss,
                 X, 
                 lasso_weights,
                 elastic_net_param=None,
                 alpha=1.  # elastic net mixing -- 1 is LASSO
                 ):

        self.saturated_loss = saturated_loss
        self.X = astransform(X)
        # the penalty parameters

        self.alpha = alpha
        self.penalty = weighted_l1norm(lasso_weights, lagrange=1)
        self.shape = self.penalty.shape
        self.group_shape = self.penalty.shape # each feature is its own group

        # elastic net part
        if elastic_net_param is None:
            elastic_net_param = np.ones(self.shape)
        self.elastic_net_param = elastic_net_param

        unpenalized = self.penalty.weights == 0
        penalized = self._penalized_vars = ~unpenalized
        self.solution = np.zeros(self.shape)
        self.ever_active = np.zeros(self.X.input_shape, np.bool)
        self.ever_active = self.updated_ever_active(unpenalized)

        if np.any(self.elastic_net_param[unpenalized]):
            warn('convention is that parameters with no l1 penalty have no Lagrange parameter in front '
                 'of their ridge term so that lambda_max is easily computed')

        if np.any(unpenalized):
            (self.final_step, 
             null_grad, 
             null_soln,
             null_linpred,
             _ ) = self.solve_subproblem(self.solution,
                                         np.nonzero(unpenalized)[0],
                                         self.BIG,
                                         tol=1.e-8)
            linear_predictor = null_linpred
            self.solution[unpenalized] = null_soln
        else:
            linear_predictor = np.zeros(self.saturated_loss.shape)

        self.grad_solution = (self.full_gradient(self.saturated_loss, 
                                                 linear_predictor) + self.enet_grad(self.solution, 
                                                                                    self._penalized_vars,
                                                                                    1))

    # LASSO specific part

    def enet_loss(self, 
                  lagrange,
                  candidate_set=None):
        """
        Ridge (elastic net) part of the objective.

        Parameters
        ----------
        lagrange : float
            Lagrange parameter.
        candidate_set : index, optional
            Restrict the ridge term to these variables.

        Returns
        -------
        loss : `regreg.smooth.quadratic.quadratic_loss`
            Diagonal quadratic loss.
        """

        return _restricted_elastic_net(self.elastic_net_param, 
                                       self.penalty.weights,
                                       lagrange,
                                       self.alpha,
                                       candidate_set)

    def subsample(self,
                  case_idx):
        '''
        Create a new path, by subsampling cases of ``self.saturated_loss``.

        Case weights are computed with ``self.saturated_loss.subsample``.

        Parameters
        ----------
        case_idx : index
            An index-like object used to specify which cases to include
            in the subsample.

        Returns
        -------
        subsample_path : path object
            A path object with a modified smooth part reflecting the
            subsampling.
        '''
        subsample_loss = self.saturated_loss.subsample(case_idx)
        return self.__class__(subsample_loss,
                              self.X,
                              self.penalty.weights,
                              elastic_net_param=self.elastic_net_param,
                              alpha=self.alpha)

    def check_KKT(self,
                  grad_solution,
                  solution,
                  lagrange,
                  penalty=None):

        '''
        Check KKT conditions over the groups in the path.

        Determines which groups are failing the KKT conditions
        (these could be active groups or inactive groups).

        Parameters
        ----------
        grad_solution : ndarray
            Candidate for gradient of smooth loss at
            Lagrange value `lagrange`.
        solution : ndarray
            Candidate for solution to problem at
            Lagrange value `lagrange`.
        lagrange : float
            Lagrange value for penalty.
        penalty : object, optional
            A weighted L1 penalty. If None, defaults to ``self.penalty``.

        Returns
        -------
        failing_active : ndarray
            Boolean indicator of active (or unpenalized) groups failing
            the KKT conditions.
        inactive_ranks : ndarray
            For inactive groups failing the KKT conditions, the rank of
            the size of the violation; -1 for all other groups.
        '''

        if penalty is None:
            penalty = self.penalty

        value = _check_KKT(penalty.weights, 
                           grad_solution, 
                           solution, 
                           lagrange)
        return value[0] > 0, value[1]

    def strong_set(self,
                   lagrange_cur,
                   lagrange_new,
                   grad_solution):
        """
        Compute the strong set at `lagrange_new`.

        Parameters
        ----------
        lagrange_cur : float
            Current Lagrange parameter.
        lagrange_new : float
            Next Lagrange parameter.
        grad_solution : ndarray
            Gradient at the solution for `lagrange_cur`.

        Returns
        -------
        strong : ndarray
            Indices of variables in the strong set.
        strong_bool : ndarray
            Boolean indicator of the strong set.
        strong_vars : ndarray
            Indices of variables in the strong set (same as `strong`).
        """

        bool = _strong_set(self.penalty.weights,
                          lagrange_cur,
                          lagrange_new,
                          grad_solution)
        cols = np.nonzero(bool)[0]
        return cols, bool, cols

    def active_set(self, solution):
        """
        Boolean indicator of nonzero coefficients in `solution`.
        """
        return solution != 0

    def restricted_penalty(self, subset):
        """
        Weighted L1 penalty restricted to `subset` (all variables if None).
        """
        if subset is not None:
            weights = self.penalty.weights[subset]
        else:
            weights = self.penalty.weights
        return weighted_l1norm(weights,
                               lagrange=1)

    def solve_subproblem(self, 
                         solution, 
                         candidate_set, 
                         lagrange_new, 
                         **solve_args):
        """
        Solve the problem restricted to a candidate set of variables.

        Parameters
        ----------
        solution : ndarray
            Current solution, used as a warm start.
        candidate_set : index
            Variables to include in the subproblem.
        lagrange_new : float
            Lagrange parameter.
        solve_args : dict
            Keyword arguments passed to ``simple_problem.solve``.

        Returns
        -------
        final_step : float
            Final step size of the solver.
        sub_grad : ndarray
            Gradient of the restricted smooth loss at `sub_soln`.
        sub_soln : ndarray
            Solution of the subproblem.
        sub_linear_pred : ndarray
            Linear predictor at `sub_soln`.
        candidate_bool : ndarray
            Boolean indicator of the candidate set.
        """
    
        # solve a problem with a candidate set

        sub_loss, sub_penalty, sub_X = _restricted_problem(self.X, 
                                                           self.saturated_loss, 
                                                           self.alpha, 
                                                           lagrange_new * self.penalty.weights, 
                                                           candidate_set,
                                                           self.subsample_columns)
        if self.alpha < 1:
            sub_elastic_net = self.enet_loss(lagrange_new,
                                             candidate_set)

            sub_loss = smooth_sum([sub_loss, sub_elastic_net])

        sub_problem = simple_problem(sub_loss, sub_penalty)
        sub_problem.coefs[:] = solution[candidate_set] # warm start
        sub_soln = sub_problem.solve(**solve_args)
        sub_grad = sub_loss.smooth_objective(sub_soln, mode='grad') 
        sub_linear_pred = sub_X.dot(sub_soln)
        candidate_bool = np.zeros(self.group_shape, np.bool)
        candidate_bool[candidate_set] = True
        return sub_problem.final_step, sub_grad, sub_soln, sub_linear_pred, candidate_bool

    def enet_grad(self,
                  solution,
                  penalized, # boolean
                  lagrange_new,
                  subset=None):
        """
        Gradient of the ridge (elastic net) part of the objective.

        Parameters
        ----------
        solution : ndarray
            Coefficients.
        penalized : ndarray
            Boolean indicator of penalized variables; for these the ridge
            term is multiplied by `lagrange_new`.
        lagrange_new : float
            Lagrange parameter.
        subset : index, optional
            Restrict to these variables.

        Returns
        -------
        G : ndarray
            Gradient of the ridge term.
        """

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
                            index_obj,
                            group_ids=False): # ignored as result is the same
        """
        Indices of variables that are ever active, with `index_obj` added.

        Parameters
        ----------
        index_obj : index
            Variables to add to the ever-active set.
        group_ids : bool, optional
            Ignored.

        Returns
        -------
        ever_active : list
            Sorted indices of ever-active variables.
        """
        if not hasattr(self, '_ever_active'):
            self._ever_active = np.zeros(self.group_shape, np.bool)
        _ever_active = self._ever_active.copy()
        _ever_active[index_obj] = True
        return list(np.nonzero(_ever_active)[0])

    @property
    def unpenalized(self):
        """
        Unpenalized variables.

        Returns
        -------
        groups : ndarray
            Indices of variables with weights equal to 0.
        variables : ndarray
            Boolean indicator that is True if no penalty on that variable.
        """
        unpen = self.penalty.weights == 0
        return np.nonzero(unpen)[0], unpen

# Some common loss factories

def logistic(X, Y, *args, **keyword_args):
    """
    Construct a `lasso_path` for the logistic regression loss with
    binary response `Y`.

    Remaining arguments are passed to `lasso_path`.
    """
    Y = np.asarray(Y)
    return lasso_path(glm.logistic_loglike(Y.shape, Y), X, *args, **keyword_args)

def probit(X, Y, *args, **keyword_args):
    """
    Construct a `lasso_path` for the probit regression loss with binary
    response `Y`.

    Remaining arguments are passed to `lasso_path`.
    """
    Y = np.asarray(Y)
    return lasso_path(glm.probit_loglike(Y.shape, Y), X, *args, **keyword_args)

def cloglog(X, Y, *args, **keyword_args):
    """
    Construct a `lasso_path` for the complementary log-log regression
    loss with binary response `Y`.

    Remaining arguments are passed to `lasso_path`.
    """
    Y = np.asarray(Y)
    return lasso_path(glm.cloglog_loglike(Y.shape, Y), X, *args, **keyword_args)

def gaussian(X, Y, *args, **keyword_args):
    """
    Construct a `lasso_path` for the Gaussian (squared error) loss with
    response `Y`.

    Remaining arguments are passed to `lasso_path`.
    """
    Y = np.asarray(Y)
    return lasso_path(glm.gaussian_loglike(Y.shape, Y), X, *args, **keyword_args)

def cox(X, T, S, *args, **keyword_args):
    """
    Construct a `lasso_path` for the Cox partial likelihood with event
    times `T` and status `S`.

    Remaining arguments are passed to `lasso_path`.
    """
    T, S = np.asarray(T), np.asarray(S)
    return lasso_path(glm.cox_loglike(T.shape, T, S), X, *args, **keyword_args)

def poisson(X, Y, *args, **keyword_args):
    """
    Construct a `lasso_path` for the Poisson regression loss with count
    response `Y`.

    Remaining arguments are passed to `lasso_path`.
    """
    Y = np.asarray(Y)
    return lasso_path(glm.poisson_loglike(Y.shape, Y), X, *args, **keyword_args)

def huber(X, Y, smoothing_parameter, *args, **keyword_args):
    """
    Construct a `lasso_path` for the Huber loss with response `Y` and
    smoothing parameter `smoothing_parameter`.

    Remaining arguments are passed to `lasso_path`.
    """
    Y = np.asarray(Y)
    return lasso_path(glm.huber_loss(Y.shape, Y, smoothing_parameter), X, *args, **keyword_args)

def huber_svm(X, Y, smoothing_parameter, *args, **keyword_args):
    """
    Construct a `lasso_path` for the smoothed (Huberized) hinge loss
    with binary response `Y` and smoothing parameter
    `smoothing_parameter`.

    Remaining arguments are passed to `lasso_path`.
    """
    Y = np.asarray(Y)
    return lasso_path(glm.huber_svm(Y.shape, Y, smoothing_parameter), X, *args, **keyword_args)

# private functions

def _get_lagrange_max(penalty,
                     grad_solution):
    """
    Smallest Lagrange parameter for which penalized coefficients are zero.
    """
    penalized = penalty.weights > 0
    return np.fabs(grad_solution[penalized] / penalty.weights[penalized]).max()


def _strong_set(lasso_weights,
                lagrange_cur,
                lagrange_new,
                gradient,
                slope_estimate=1):
    """
    Return a Boolean indicator for each group of whether it is in the
    strong set or not.
    """
    thresh = 2 * lagrange_new - lagrange_cur
    scaled_grad = np.fabs(gradient)
    scaled_grad[lasso_weights > 0] /= lasso_weights[lasso_weights > 0]
    scaled_grad[lasso_weights == 0] = np.inf
    return scaled_grad > thresh

def _restricted_elastic_net(elastic_net_params, 
                            lasso_weights, 
                            lagrange, 
                            alpha, 
                            candidate_set=None):
    penalized = lasso_weights != 0
    new_params = elastic_net_params * (1 - alpha)
    new_params[penalized] *= lagrange 
    if candidate_set is not None:
        new_params = new_params[candidate_set]
    return quadratic_loss(new_params.shape,
                          new_params,
                          Qdiag=True)

def _restricted_problem(X, 
                        saturated_loss, 
                        alpha,
                        lasso_weights, 
                        candidate_set, 
                        subsample_columns):

    X_candidate = subsample_columns(X, candidate_set)
    restricted_loss = affine_smooth(saturated_loss, X_candidate)
    restricted_penalty = weighted_l1norm(lasso_weights[candidate_set] * alpha, lagrange=1.)

    return restricted_loss, restricted_penalty, X_candidate

def _check_KKT(lasso_weights, 
               grad, 
               solution, 
               lagrange, 
               tol=1.e-2):

    """
    Check whether ``(grad, solution)`` satisfy KKT conditions at a given
    tolerance.
    """

    ACTIVE = 1
    INACTIVE = 2
    UNPENALIZED = 3

    active = (solution != 0) * lasso_weights > 0
    active_failing = np.zeros(solution.shape, np.int)
    active_failing[active] = (np.fabs(grad[active] / (lagrange * lasso_weights[active]) + np.sign(solution[active])) > tol) * ACTIVE

    # inactive_failing = np.zeros(solution.shape, np.int) - 1
    # 
    # inactive_failing[inactive] = (np.fabs(grad[inactive] / (lagrange * lasso_weights[inactive])) > 1 + tol) * INACTIVE

    inactive = (solution == 0) * lasso_weights > 0
    inactive_terms = np.zeros(solution.shape)
    inactive_terms[inactive] = np.fabs(grad[inactive] / (lagrange * lasso_weights[inactive]))
    inactive_ranks = solution.shape[0] - rankdata(inactive_terms)
    inactive_ranks[inactive_terms <= 1+tol] = -1

    unpenalized = lasso_weights == 0
    active_failing[unpenalized] = (np.fabs(grad[unpenalized] / np.mean(lasso_weights[lasso_weights>0])) > tol) * UNPENALIZED

    np.testing.assert_allclose(inactive_ranks >= 0, inactive_terms > 1 + tol)

    return active_failing, inactive_ranks
