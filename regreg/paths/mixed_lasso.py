"""
Regularization paths for the mixed LASSO penalty.

This module uses an older path API than the other path modules.
"""

from __future__ import print_function, division, absolute_import

from copy import copy
from warnings import warn
import gc

import numpy as np
import numpy.linalg as npl

from .lasso import lasso_path

import scipy.sparse

from ..affine import power_L, normalize, selector, identity, add_intercept
from ..atoms.seminorms import l1norm, constrained_positive_part
from ..smooth import sum as smooth_sum, affine_smooth
from ..smooth.glm import glm
from ..smooth.quadratic import quadratic_loss
from ..problems.simple import simple_problem
from ..identity_quadratic import identity_quadratic as iq
from ..atoms.mixed_lasso import mixed_lasso, strong_set as strong_set_ml, check_KKT

# Constants used below

UNPENALIZED = -1
L1_PENALTY = -2
POSITIVE_PART = -3
NONNEGATIVE = -4

def _column_scales(X, center, ddof=1):
    """
    Column scales used by `regreg.affine.normalize`.

    With `center`, these are the sample standard deviations (with `ddof`);
    otherwise the root mean squares of the columns.
    """
    n = X.shape[0]
    if scipy.sparse.issparse(X):
        sum_sq = np.asarray(X.multiply(X).sum(0)).reshape(-1)
        means = np.asarray(X.mean(0)).reshape(-1)
    else:
        sum_sq = np.sum(X**2, 0)
        means = X.mean(0)
    if center:
        return np.sqrt(np.maximum(sum_sq - n * means**2, 0) / (n - ddof))
    return np.sqrt(sum_sq / (n - (ddof - 1)))

class _mixed_lasso_design(object):
    """
    The design matrix used by `mixed_lasso_path`.

    Its columns are an optional intercept column (never centered or
    scaled) followed by the columns of `X`, centered and/or scaled as
    requested. When centering or scaling, constant columns of `X` (zero
    scale) are dropped; `keep` marks the columns of `X` that are kept.

    Parameters
    ----------
    X : ndarray or sparse matrix
        Design matrix.
    intercept : bool
        Prepend an intercept column?
    center : bool
        Center the columns of `X`?
    scale : bool
        Scale the columns of `X`?
    """

    def __init__(self, X, intercept, center, scale):
        if scipy.sparse.issparse(X):
            X = scipy.sparse.csc_matrix(X)
        else:
            X = np.asarray(X)
        self.intercept, self.center, self.scale = intercept, center, scale

        n, p = X.shape
        if center or scale:
            col_scales = _column_scales(X, center)
            self.keep = col_scales > 0
        else:
            col_scales = np.ones(p)
            self.keep = np.ones(p, bool)
        self._X = X[:, self.keep]
        self._col_scales = col_scales[self.keep]
        self.shape = (n, int(self.keep.sum()) + int(intercept))

    @property
    def scalings(self):
        """
        Scale of each column of the design: dividing coefficients for
        the normalized design by these gives coefficients for the
        original columns.
        """
        scalings = self._col_scales if self.scale else np.ones(self._X.shape[1])
        if self.intercept:
            scalings = np.hstack([1, scalings])
        return scalings

    def columns(self, which=None):
        """
        The (normalized) design restricted to some of its columns.

        Parameters
        ----------
        which : index, optional
            Boolean mask or sorted indices of the columns of the design
            to keep. Defaults to all columns.

        Returns
        -------
        design : ndarray, sparse matrix or `regreg.affine.affine_transform`
        """
        idx = np.arange(self.shape[1])
        if which is not None:
            idx = idx[which]
        has_intercept = self.intercept and np.any(idx == 0)
        xcols = idx[idx >= int(self.intercept)] - int(self.intercept)

        n = self.shape[0]
        Xs = self._X[:, xcols]
        normalizing = self.center or self.scale
        if normalizing and xcols.shape[0] > 0:
            Xs = normalize(Xs, center=self.center, scale=self.scale)

        if not has_intercept:
            return Xs
        if xcols.shape[0] == 0:
            return np.ones((n, 1))
        if normalizing:
            return add_intercept(Xs)
        if scipy.sparse.issparse(Xs):
            return scipy.sparse.hstack([np.ones((n, 1)), Xs]).tocsc()
        return np.hstack([np.ones((n, 1)), Xs])

class mixed_lasso_path(lasso_path):

    """
    Regularization path for the mixed LASSO penalty.

    The penalty, `regreg.atoms.mixed_lasso.mixed_lasso`, allows each feature
    to be unpenalized, L1 penalized, positive part penalized, constrained to
    be nonnegative, or to belong to a group penalized by its L2 norm.

    Parameters
    ----------
    loss_factory : callable
        Function that takes a (normalized) design and returns the smooth
        loss.
    X : ndarray or sparse matrix
        Design matrix.
    penalty_structure : ndarray, optional
        Penalty type for each feature: one of the module constants
        `UNPENALIZED`, `L1_PENALTY`, `POSITIVE_PART`, `NONNEGATIVE`, or a
        nonnegative group label. Defaults to `L1_PENALTY` for all features.
    group_weights : dict, optional
        Weights for the groups implied by `penalty_structure`.
    elastic_net : `regreg.identity_quadratic.identity_quadratic`, optional
        Elastic net quadratic.
    alpha : float, optional
        Elastic net mixing parameter.
    intercept : bool, optional
        Add an unpenalized intercept column?
    positive_part : optional
        Unused.
    unpenalized : optional
        Unused.
    lagrange_proportion : float, optional
        Ratio of the smallest to the largest Lagrange parameter.
    nstep : int, optional
        Number of Lagrange parameters.
    scale : bool, optional
        Scale the columns of `X`?
    center : bool, optional
        Center the columns of `X`?
    """

    def __init__(self, 
                 loss_factory, 
                 X, 
                 penalty_structure=None, 
                 group_weights={},
                 elastic_net=iq(0,0,0,0),
                 alpha=0., intercept=True,
                 positive_part=None,
                 unpenalized=None,
                 lagrange_proportion = 0.05,
                 nstep = 100,
                 scale=True,
                 center=True):


        self.loss_factory = loss_factory

        self.scale = scale
        self.center = center

        # for group lasso weights, if implied by penalty_structure
        self.group_weights = group_weights

        # penalty structure, with an unpenalized intercept if needed
        self.intercept = intercept
        p = X.shape[1]
        if self.intercept:
            penalty_structure_full = np.ones(p+1) * L1_PENALTY
            penalty_structure_full[0] = UNPENALIZED
            if penalty_structure is not None:
                penalty_structure_full[1:] = penalty_structure
        else:
            penalty_structure_full = np.ones(p) * L1_PENALTY
            if penalty_structure is not None:
                penalty_structure_full[:] = penalty_structure

        # normalize X, dropping constant columns
        self._design = _mixed_lasso_design(X, self.intercept, self.center, self.scale)
        keep = self._design.keep
        if self.intercept:
            keep = np.hstack([True, keep])
        if np.all(keep):
            self._selector = identity((keep.shape[0],))
        else:
            self._selector = selector(keep, (keep.shape[0],))
        self.penalty_structure = penalty_structure_full[keep]
        self._Xn = self._design.columns()

        # the penalty parameters
        self.alpha = alpha
        self.lagrange_proportion = lagrange_proportion
        self.nstep = nstep
        self._elastic_net = elastic_net.collapsed()

        self.initial_active = (np.equal(self.penalty_structure, UNPENALIZED) + 
                               np.equal(self.penalty_structure, NONNEGATIVE))
        self.ever_active = self.initial_active.copy()

    @property
    def shape(self):
        """
        Shape ``(n, p)`` of the normalized design.
        """
        return self._design.shape

    @property
    def nonzero(self):
        """
        Selector for the columns with nonzero standard deviation.
        """
        return self._selector

    @property
    def elastic_net(self):
        """
        Elastic net quadratic scaled by the current Lagrange parameter.
        """
        q = copy(self._elastic_net)
        q.coef *= self.lagrange
        q.linear_term *= self.lagrange
        return q

    @property
    def Xn(self):
        """
        The normalized design.
        """
        return self._Xn

    @property
    def loss(self):
        """
        Smooth loss for the normalized design.
        """
        if not hasattr(self, '_loss'):
            self._loss = self.loss_factory(self._Xn)
        return self._loss

    @property
    def null_solution(self):
        """
        Solution with only the initially active (unpenalized or
        nonnegative) features.
        """
        if not hasattr(self, "_null_soln"):
            n, p = self.shape
            self._null_soln = np.zeros(p)
            null_problem, null_selector = self.restricted_problem(self.initial_active, self.lagrange_max)[:2]
            self._null_soln = null_selector.adjoint_map(null_problem.solve())
        return self._null_soln

    @property
    def lagrange_max(self):
        """
        Smallest Lagrange parameter for which the solution is the null
        solution.
        """
        if not hasattr(self, "_lagrange_max"):
            null_soln = self.null_solution
            null_grad = self.loss.smooth_objective(null_soln, 'grad')
            self.penalty = mixed_lasso(self.penalty_structure, 1., weights=self.group_weights)
            conj = self.penalty.conjugate
            self._lagrange_max = conj.seminorm(null_grad)

        return self._lagrange_max

    def get_lagrange_sequence(self):
        """
        Decreasing, log-spaced sequence of Lagrange parameters.
        """
        if not hasattr(self, "_lagrange_sequence"):
            self._lagrange_sequence = self.lagrange_max * np.exp(np.linspace(np.log(self.lagrange_proportion), 0, 
                                                                             self.nstep))[::-1]
        return self._lagrange_sequence

    def set_lagrange_sequence(self, lagrange_sequence):
        """
        Set the sequence of Lagrange parameters.
        """
        self._lagrange_sequence = lagrange_sequence
    
    lagrange_sequence = property(get_lagrange_sequence, set_lagrange_sequence)

    @property
    def problem(self):
        """
        The full problem, with all features.
        """
        p = self.shape[1]
        if not hasattr(self, "_problem"):
            self._problem = self.restricted_problem(np.ones(self.shape[1], bool),
                                                    self.lagrange_max)[0]
        return self._problem

    def get_lagrange(self):
        """
        Current Lagrange parameter of the full problem.
        """
        return self._problem.proximal_atom.lagrange

    def set_lagrange(self, lagrange):
        """
        Set the Lagrange parameter of the full problem and the penalty.
        """
        proximal_atom = self._problem.proximal_atom
        proximal_atom.lagrange = lagrange
        self.penalty.lagrange = lagrange
    lagrange = property(get_lagrange, set_lagrange)

    @property
    def solution(self):
        """
        Current coefficients of the full problem.
        """
        return self.problem.coefs

    @property
    def active(self):
        """
        Boolean indicator of nonzero coefficients.
        """
        return self.solution != 0

    @property
    def lipschitz(self):
        """
        Estimate of the largest squared singular value of the normalized
        design.
        """
        if not hasattr(self, "_lipschitz"):
            self._lipschitz = power_L(self.Xn)
        return self._lipschitz

    def grad(self, loss=None):
        '''
        Gradient at current value.

        This includes the gradient of the smooth loss as well as the
        gradient of the elastic net part. This is used for determining
        whether the KKT conditions are met and which coefficients are in
        the strong set.

        Parameters
        ----------
        loss : `regreg.smooth.smooth_atom`, optional
            Smooth loss. Defaults to ``self.loss``.

        Returns
        -------
        grad : ndarray
            Gradient at ``self.solution``.
        '''
        if loss is None:
            loss = self.loss
        gsmooth = loss.smooth_objective(self.solution, 'grad')
        penalized = self.penalty_structure != UNPENALIZED
        # XXX the elastic net is probably not quite right here if the elastic net has a non-zero center
        gquad = self.elastic_net.objective(self.solution[penalized], 'grad')
        gsmooth[penalized] += gquad

        return gsmooth

    def strong_set(self, lagrange_cur, lagrange_new, grad=None,
                   slope_estimate=1):
        """
        Compute the strong set at `lagrange_new`.

        Parameters
        ----------
        lagrange_cur : float
            Current Lagrange parameter.
        lagrange_new : float
            Next Lagrange parameter.
        grad : ndarray, optional
            Gradient at the current solution. Defaults to ``self.grad()``.
        slope_estimate : float, optional
            Estimate of the slope used in the strong rule.

        Returns
        -------
        strong : ndarray
            Boolean indicator of the strong set.
        strong_selector : `regreg.affine.selector`
            Selector for the strong set.
        """
        if grad is None:
            grad = self.grad()

        return strong_set_ml(self.penalty, lagrange_cur, lagrange_new, grad, slope_estimate)

    def slice_columns(self, columns):
        """
        Columns of the normalized design.
        """
        return self._design.columns(columns)

    def construct_loss(self, candidate_set, lagrange):
        """
        Construct the loss restricted to `candidate_set`.

        Parameters
        ----------
        candidate_set : index
            Columns to include.
        lagrange : float
            Lagrange parameter (unused).

        Returns
        -------
        Xslice : object
            Restricted design.
        loss : `regreg.smooth.smooth_atom`
            Restricted loss.
        """
        Xslice = self.slice_columns(candidate_set)
        loss = self.loss_factory(Xslice)
        return Xslice, loss

    def restricted_problem(self, candidate_set, lagrange):
        '''
        Construct the problem restricted to `candidate_set`.

        Assumes the candidate set includes intercept as first column.

        Parameters
        ----------
        candidate_set : index
            Columns to include.
        lagrange : float
            Lagrange parameter.

        Returns
        -------
        problem : `regreg.problems.simple.simple_problem`
            Restricted problem.
        candidate_selector : `regreg.affine.selector`
            Selector for `candidate_set`.
        penalty_structure : ndarray
            Penalty structure restricted to `candidate_set`.
        '''

        Xslice, loss = self.construct_loss(candidate_set, lagrange)

        restricted_penalty_structure = self.penalty_structure[candidate_set]
        rps = restricted_penalty_structure # shorthand

        sliced_penalty = mixed_lasso(rps, lagrange, weights=self.group_weights)
        # the elastic net term lagrange * coef / 2 ||beta_penalized||^2, as in
        # `grad` and the KKT checks (it was left out of the problems solved)
        ridge = (self._elastic_net.coef or 0) * lagrange
        if ridge != 0:
            ridge_weights = ridge * (rps != UNPENALIZED)
            loss = smooth_sum([loss, quadratic_loss(ridge_weights.shape,
                                                    ridge_weights,
                                                    Qdiag=True)])
        problem_sliced = simple_problem(loss, sliced_penalty)
        candidate_selector = selector(candidate_set, self.shape[1])
        return problem_sliced, candidate_selector, restricted_penalty_structure

    def solve_subproblem(self, candidate_set, lagrange_new, **solve_args):
        """
        Solve the problem restricted to `candidate_set`.

        The solution is stored in ``self.solution``.

        Parameters
        ----------
        candidate_set : index
            Columns to include.
        lagrange_new : float
            Lagrange parameter.
        solve_args : dict
            Keyword arguments passed to ``simple_problem.solve``.

        Returns
        -------
        final_step : float
            Final step size of the solver.
        grad : ndarray
            Gradient of the restricted smooth loss at the solution.
        sub_soln : ndarray
            Solution of the subproblem.
        penalty_structure : ndarray
            Penalty structure restricted to `candidate_set`.
        """
    
        # try to solve the problem with the active set
        subproblem, selector, penalty_structure = self.restricted_problem(candidate_set, lagrange_new)
        subproblem.coefs[:] = selector.linear_map(self.solution)
        sub_soln = subproblem.solve(**solve_args)
        self.solution[:] = selector.adjoint_map(sub_soln)

        grad = subproblem.smooth_objective(sub_soln, mode='grad') 
        self.final_step = subproblem.final_step
        return self.final_step, grad, sub_soln, penalty_structure

    def main(self, inner_tol=1.e-5, verbose=False, kkt_tol=None):
        """
        Compute the path over ``self.lagrange_sequence``.

        Parameters
        ----------
        inner_tol : float, optional
            Tolerance for each subproblem.
        verbose : bool, optional
            Print progress information?
        kkt_tol : float, optional
            Relative tolerance of the KKT check for coordinates and groups
            at zero. If None, ``self.kkt_tol`` is used.

        Returns
        -------
        output : dict
            Dictionary with keys ``'devratio'``, ``'df'``, ``'lagrange'``,
            ``'scalings'`` and ``'beta'``.
        """

        # scaling will be needed to get coefficients on original scale   
        # (a copy: selector.adjoint_map reuses its output buffer)
        scalings = self.nonzero.adjoint_map(self._design.scalings).copy()

        # take a guess at the inverse step size
        self.final_step = 1000. / self.lipschitz 
        lseq = self.lagrange_sequence # shorthand

        # first solution corresponding to all zeros except intercept 

        self.solution[:] = self.null_solution.copy()

        grad_solution = self.grad().copy()
        strong, strong_selector = self.strong_set(lseq[0], lseq[1], grad=grad_solution)

        # coefficients for the normalized design, like the later rows
        rescaled_solutions = scipy.sparse.csr_matrix(self.nonzero.adjoint_map(self.solution))

        objective = [self.loss.smooth_objective(self.solution, 'func')]
        # not quite right -- should check tight constraints
        dfs = [np.sum(self.initial_active)]
        retry_counter = 0

        all_failing = np.zeros(grad_solution.shape, bool)
        if kkt_tol is None:
            kkt_tol = self.kkt_tol

        for lagrange_new, lagrange_cur in zip(lseq[1:], lseq[:-1]):
            self.lagrange = lagrange_new
            tol = inner_tol
            active_old = self.active.copy()
            num_tries = 0
            debug = False
            coef_stop = True
            while True:
                strong, strong_selector = self.strong_set(lagrange_cur, 
                                                          lagrange_new, grad=grad_solution)

                subproblem_set = self.ever_active + all_failing
                final_step, grad, sub_soln, penalty_structure \
                    = self.solve_subproblem(subproblem_set,
                                            lagrange_new,
                                            tol=tol,
                                            start_step=self.final_step,
                                            debug=debug and verbose,
                                            coef_stop=coef_stop)

                p = self.shape[1]

                self.solution[subproblem_set] = sub_soln
                # this only corrects the gradient on the subproblem_set
                grad_solution[subproblem_set] = grad

                strong_problem = self.restricted_problem(strong, lagrange_new)[0]
                strong_soln = self.solution[strong]
                # the restricted problem's smooth part includes the elastic net
                strong_grad = strong_problem.smooth_objective(strong_soln, mode='grad')
                strong_penalty = strong_problem.proximal_atom

                strong_failing = check_KKT(strong_penalty, strong_grad, strong_soln, lagrange_new,
                                           kkt_tol=kkt_tol)

                if np.any(strong_failing):
                    all_failing += (
                        strong_selector.adjoint_map(strong_failing) != 0)
                else:
                    self.solution[subproblem_set] = sub_soln
                    grad_solution = self.grad()
                    all_failing = check_KKT(self.penalty, grad_solution, self.solution, lagrange_new,
                                            kkt_tol=kkt_tol)

                    if not all_failing.sum():
                        self.ever_active += self.solution != 0
                        self.final_step = final_step
                        break
                    else:
                        if verbose:
                            print('failing:', np.nonzero(all_failing)[0])
                        retry_counter += 1
                        self.ever_active += all_failing

                tol /= 2.
                num_tries += 1
                if num_tries % 5 == 0:

                    self.solution[subproblem_set] = sub_soln
                    self.solution[~subproblem_set] = 0
                    grad_solution = self.grad()

                    debug = True
                    tol = inner_tol
                    if num_tries >= 10:
                        warn('convergence not achieved for lagrange=%0.4e' % lagrange_new)
                        break

            rescaled_solution = self.nonzero.adjoint_map(self.solution)
            rescaled_solutions = scipy.sparse.vstack([rescaled_solutions, rescaled_solution])
            objective.append(self.loss.smooth_objective(self.solution, mode='func'))
            dfs.append((self.solution != 0).sum())
            gc.collect()

            if verbose:
                print(lagrange_cur / self.lagrange_max,
                      lagrange_new,
                      (self.solution != 0).sum(),
                      1. - objective[-1] / objective[0],
                      list(self.lagrange_sequence).index(lagrange_new),
                      np.fabs(rescaled_solution).sum())

        objective = np.array(objective)
        output = {'devratio': 1 - objective / objective.max(),
                  'df': dfs,
                  'lagrange': self.lagrange_sequence,
                  'scalings': scalings,
                  'beta':rescaled_solutions.T}

        return output

    # Some common loss factories

    @classmethod
    def logistic(cls, X, Y, *args, **keyword_args):
        """
        Construct a path for the logistic loss with binary response `Y`.
        """
        return cls(lambda Xs: glm.logistic(Xs, Y), X, *args, **keyword_args)

    @classmethod
    def gaussian(cls, X, Y, *args, **keyword_args):
        """
        Construct a path for the Gaussian loss with response `Y`.
        """
        return cls(lambda Xs: glm.gaussian(Xs, Y), X, *args, **keyword_args)

