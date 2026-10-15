"""
Generic machinery for computing regularization paths.

This module defines the `grouped_path` base class, which specific path
classes (LASSO, group LASSO, etc.) subclass, as well as the path algorithms
`strong_rules` and `warm_start` that operate on such objects.
"""

import gc
from warnings import warn
import numpy as np

from ..affine import astransform, power_L
from ..smooth import affine_smooth, sum as smooth_sum
from ..problems.simple import simple_problem

def subsample_columns(X, columns):
    """
    Extract columns of a (possibly implicit) design matrix.

    Parameters
    ----------
    X : ndarray or `regreg.affine.affine_transform`
        Design matrix with one-dimensional input shape.
    columns : sequence of int
        Columns to extract.

    Returns
    -------
    cols : ndarray
        Array of shape ``X.output_shape + (len(columns),)`` holding the
        selected columns.
    """

    X = astransform(X)
    cols = np.zeros((len(columns),) + X.output_shape)
    indicator = np.zeros(X.input_shape[0]) # assuming 1-dimensional input and output shape here
    for i, col in enumerate(columns):
        indicator[col] = 1 # 1-hot vector
        cols[i] = X.dot(indicator)
        indicator[col] = 0 # back to 0 vector
    return cols.T

class grouped_path(object):

    """
    Base class for regularization paths over groups of variables.

    Subclasses implement the methods used by `strong_rules` and
    `warm_start` (e.g. `check_KKT`, `strong_set`, `solve_subproblem`).
    """

    BIG = 1e12 # lagrange parameter for finding null solution

    # relative tolerance of the KKT check for groups at zero: a group whose
    # dual norm exceeds lagrange * (1 + kkt_tol) enters the subproblem.
    # Looser values let a group at the edge of the active set be left out
    # (gh-99). The checks of active groups keep their own, looser, tolerance
    # as they depend on the accuracy of the inner solves.
    kkt_tol = 1.e-6

    def __init__(self, 
                 saturated_loss,
                 X, 
                 lasso_weights,
                 elastic_net_param=None,
                 alpha=1.  # elastic net mixing -- 1 is LASSO
                 ):
        raise NotImplementedError

    # used for warm start path

    def full_loss(self):
        """
        Saturated loss composed with the design matrix ``self.X``.
        """
        loss = affine_smooth(self.saturated_loss,
                             self.X)
        loss.shape = self.X.input_shape
        return loss

    def enet_loss(self, 
                  lagrange,
                  candidate_set):
        raise NotImplementedError

    # methods potentially overwritten in subclasses for I/O considerations

    def subsample_columns(self, 
                          X, 
                          columns):
        """
        Extract columns of X into ndarray or regreg transform.

        Parameters
        ----------
        X : object
            Representation of design matrix.
        columns : sequence of int
            Columns to extract.

        Returns
        -------
        cols : ndarray or `regreg.affine.affine_transform`
            The selected columns.
        """
        return subsample_columns(X, 
                                 columns)

    def full_gradient(self, 
                      saturated_loss, 
                      linear_predictor):
        """
        Gradient of saturated loss composed with ``self.X``.

        Parameters
        ----------
        saturated_loss : `regreg.smooth.smooth_atom`
            Loss as a function of the linear predictor.
        linear_predictor : ndarray
            Linear predictor at which to evaluate the gradient.

        Returns
        -------
        gradient : ndarray
            Gradient with respect to the coefficients.
        """
        saturated_grad = saturated_loss.smooth_objective(linear_predictor, 'grad')
        return self.X.T.dot(saturated_grad)

    def linpred(self,
                coef,
                X,
                test):
        """
        Form linear predictor for given set of coefficients.

        Typical use case will have coef of shape ``(l,p)`` for
        univariate response regressions and ``(l,q,p)`` for
        multiple response regressions (e.g. multinomial with
        ``q`` classes) where ``l`` is the number of Lagrange
        parameters.

        Parameters
        ----------
        coef : ndarray
            A set of coefficients.
        X : object
            Representation of design matrix, usually ``self.X`` -- subclasses
            may not assume this is actually an array.
        test : index
            Indices of ``X.dot(coef.T)`` to return.

        Returns
        -------
        linpred : ndarray
            Matrix of linear predictors.
        """
        coef = np.asarray(coef)
        if coef.ndim > 1:
            coefT = coef.T
            old_shape = coefT.shape
            coefT = coefT.reshape((coefT.shape[0], -1))
            linpred = X.dot(coefT)[test]
            linpred = linpred.reshape((linpred.shape[0],) + old_shape[1:])
            return linpred
        return X.dot(coef.T)[test]

    # method potentially overwritten in subclasses for penalty considerations

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
        raise NotImplementedError

    def check_KKT(self,
                  grad_solution,
                  solution,
                  lagrange,
                  penalty=None,
                  kkt_tol=None):

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
            A penalty. If None, defaults to ``self.penalty``.
        kkt_tol : float, optional
            Relative tolerance for the groups at zero: such a group fails
            if its dual norm exceeds ``lagrange * (1 + kkt_tol)``. If None,
            defaults to ``self.kkt_tol``.

        Returns
        -------
        failing_A : ndarray
            Boolean indicator of active groups failing the KKT conditions.
        failing_I_ranks : ndarray
            For inactive groups, the rank of each failing group (by the
            size of its violation), with -1 for groups that do not fail.
        '''
        raise NotImplementedError

    def strong_set(self,
                   lagrange_cur,
                   lagrange_new,
                   grad_solution):
        raise NotImplementedError

    def active_set(self, 
                   solution):
        raise NotImplementedError

    def restricted_penalty(self, 
                           subset):
        raise NotImplementedError

    def solve_subproblem(self, 
                         candidate_set, 
                         lagrange_new, 
                         **solve_args):
        raise NotImplementedError

    def enet_grad(self,
                  solution,
                  penalized, # boolean
                  lagrange_new,
                  subset=None):
        raise NotImplementedError

    def updated_ever_active(self,
                            index_obj):
        raise NotImplementedError

def default_lagrange_sequence(penalty,
                              null_solution,
                              lagrange_proportion=0.05,
                              nstep=100,
                              alpha=1):
    """
    Default decreasing, log-spaced sequence of Lagrange parameters.

    Parameters
    ----------
    penalty : `regreg.atoms.atom`
        Penalty for the path.
    null_solution : ndarray
        Gradient of the loss at the null solution. The largest Lagrange
        parameter is the dual seminorm of this value.
    lagrange_proportion : float, optional
        Ratio of the smallest to the largest Lagrange parameter.
    nstep : int, optional
        Number of Lagrange parameters.
    alpha : float, optional
        Elastic net mixing parameter; the sequence is divided by `alpha`.

    Returns
    -------
    lagrange_seq : ndarray
        Decreasing sequence of Lagrange parameters.
    """
    dual = penalty.conjugate
    lagrange_max = dual.seminorm(null_solution, lagrange=1)
    return (lagrange_max * np.exp(np.linspace(np.log(lagrange_proportion), 
                                              0, 
                                              nstep)))[::-1] / alpha
   
def strong_rules(path_obj,
                 lagrange_seq,
                 initial_data, # (solution, grad) pair
                 inner_tol=1.e-5,
                 verbose=False,
                 initial_step=None,
                 check_active=False,
                 max_tries=20,
                 kkt_tol=None):
    """
    Compute a regularization path using strong rules screening.

    Parameters
    ----------
    path_obj : `grouped_path`
        Path object describing the problem.
    lagrange_seq : sequence of float
        Decreasing sequence of Lagrange parameters.
    initial_data : tuple
        Pair ``(solution, grad_solution)`` at ``lagrange_seq[0]``.
    inner_tol : float, optional
        Tolerance for each subproblem.
    verbose : bool, optional
        Print progress information?
    initial_step : float, optional
        Initial step size for the solver. If None, it is estimated from the
        Lipschitz constant of ``path_obj.X``.
    check_active : bool, optional
        Also check KKT conditions for active groups?
    max_tries : int, optional
        Maximum number of attempts at each Lagrange parameter.
    kkt_tol : float, optional
        Relative tolerance of the KKT check for groups at zero. If None,
        ``path_obj.kkt_tol`` is used.

    Returns
    -------
    output : dict
        Dictionary with keys ``'dev explained'``, ``'lagrange'`` and
        ``'beta'`` (the solutions along the path).
    """

    # take a guess at the inverse step size
    if initial_step is None:
        _lipschitz = power_L(path_obj.X, max_its=50)
        final_step = 1000. / _lipschitz 
    else:
        final_step = initial_step

    # gradient of restricted elastic net at lambda_max

    solution, grad_solution = initial_data

    solution = solution.copy()
    grad_solution = grad_solution.copy()

    linear_predictor = path_obj.X.dot(solution)

    # unpenalized groups are in every subproblem, as in `basil`;
    # the KKT checks for inactive groups never add them
    unpen = np.zeros(path_obj.group_shape, bool)
    unpen[path_obj.unpenalized[1]] = True
    ever_active = path_obj.updated_ever_active(unpen)

    dev_solution = path_obj.saturated_loss.smooth_objective(linear_predictor, 'func')
    solutions = [solution.T.copy()]
    dev_explained = [dev_solution]

    all_failing = np.zeros(path_obj.group_shape, bool)
    subproblem_set = path_obj.updated_ever_active(unpen)

    for lagrange_new, lagrange_cur in zip(lagrange_seq[1:], 
                                          lagrange_seq[:-1]):
        tol = inner_tol
        num_tries = 0
        debug = False
        coef_stop = True

        while True:

            subproblem_set = sorted(set(subproblem_set + path_obj.updated_ever_active(all_failing)))

            (final_step, 
             subproblem_grad, 
             subproblem_soln,
             subproblem_linpred,
             subproblem_vars) = path_obj.solve_subproblem(solution, # for warm start
                                                          subproblem_set,
                                                          lagrange_new,
                                                          tol=tol,
                                                          start_step=final_step,
                                                          debug=debug and verbose,
                                                          coef_stop=coef_stop)

            saturated_grad = path_obj.saturated_loss.smooth_objective(subproblem_linpred, 'grad')
            # as subproblem always contains ever active, 
            # rest of solution should be 0
            solution[subproblem_vars] = subproblem_soln

            # strong rules step
            # a set of group ids

            strong, strong_idx, strong_vars = path_obj.strong_set(lagrange_cur * path_obj.alpha, 
                                                                  lagrange_new * path_obj.alpha, 
                                                                  grad_solution)
            strong_enet_grad = path_obj.enet_grad(solution,
                                                  path_obj._penalized_vars,
                                                  lagrange_new,
                                                  subset=strong_vars)
            strong_soln = solution[strong_vars]
            X_strong = path_obj.subsample_columns(path_obj.X, 
                                                  strong_vars)
            strong_grad = (X_strong.T.dot(saturated_grad) +
                           strong_enet_grad)
            strong_penalty = path_obj.restricted_penalty(strong_vars)
            (strong_A,
             strong_I_ranks) = path_obj.check_KKT(strong_grad, 
                                                  strong_soln, 
                                                  path_obj.alpha * lagrange_new, 
                                                  penalty=strong_penalty,
                                                  kkt_tol=kkt_tol)

            if check_active:
                strong_failing = strong_A + (strong_I_ranks >= 0)
            else:
                strong_failing = strong_I_ranks >= 0

            if np.any(strong_failing):
                delta = np.zeros(path_obj.group_shape, bool)
                delta[strong_idx] = strong_failing
                all_failing += delta 
            else:
                enet_grad = path_obj.enet_grad(solution, 
                                               path_obj._penalized_vars,
                                               lagrange_new)
                grad_solution[:] = (path_obj.full_gradient(path_obj.saturated_loss, 
                                                           subproblem_linpred) + 
                                    enet_grad)
                failing_A, failing_I_ranks = path_obj.check_KKT(grad_solution, 
                                                                solution, 
                                                                path_obj.alpha * lagrange_new,
                                                                kkt_tol=kkt_tol)

                if check_active:
                    all_failing = failing_A + (failing_I_ranks >= 0)
                else:
                    all_failing = failing_I_ranks >= 0

                if not all_failing.sum():
                    path_obj.ever_active = path_obj.updated_ever_active(path_obj.active_set(solution))
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

                enet_grad = path_obj.enet_grad(solution, 
                                               path_obj._penalized_vars,
                                               lagrange_new)
                grad_solution[:] = (path_obj.full_gradient(path_obj.saturated_loss, 
                                                           subproblem_linpred) + 
                                    enet_grad)
                debug = True
                tol = inner_tol
                if num_tries >= max_tries:
                    warn('convergence of active set not achieved for lagrange=%0.4e, moving on' % lagrange_new)
                    break

            subproblem_set = sorted(set(subproblem_set + path_obj.updated_ever_active(all_failing)))

        solutions.append(solution.T.copy())
        dev_explained.append(path_obj.saturated_loss.smooth_objective(linear_predictor, mode='func'))
        gc.collect()

        if verbose:
            print({'lagrange':lagrange_new,
                   'sparsity':(solution != 0).sum(),
                   'dev explained':1. - dev_explained[-1] / dev_explained[0],
                   'point on path':list(lagrange_seq).index(lagrange_new),
                   'penalty':path_obj.penalty.seminorm(solution, lagrange=1),
                   'dual penalty':path_obj.penalty.conjugate.seminorm(grad_solution, lagrange=1)})

    dev_explained = np.array(dev_explained)
    output = {'dev explained': 1 - dev_explained[-1] / dev_explained[0],
              'lagrange': lagrange_seq,
              'beta':np.array(solutions)}

    return output

def warm_start(path_obj,
               lagrange_seq,
               initial_data,
               inner_tol=1.e-5,
               verbose=False,
               initial_step=None,
               solve_args=None):
    """
    Compute a regularization path using warm starts.

    Each problem along the path is solved with all variables, starting
    from the solution at the previous Lagrange parameter.

    Parameters
    ----------
    path_obj : `grouped_path`
        Path object describing the problem.
    lagrange_seq : sequence of float
        Sequence of Lagrange parameters.
    initial_data : tuple
        Pair ``(solution, grad_solution)``; only `solution` is used, as the
        initial warm start.
    inner_tol : float, optional
        Tolerance for each problem.
    verbose : bool, optional
        Print progress information?
    initial_step : float, optional
        Initial step size for the solver. If None, it is estimated from the
        Lipschitz constant of ``path_obj.X``.
    solve_args : dict, optional
        Extra arguments passed to ``simple_problem.solve``.

    Returns
    -------
    output : dict
        Dictionary with keys ``'devratio'``, ``'lagrange'`` and ``'beta'``
        (the solutions along the path).
    """

    solution, _ = initial_data
    solve_args = dict(solve_args or {})

    # basic setup

    loss = path_obj.full_loss()
    all_vars = np.ones(path_obj.penalty.shape, bool)
    penalty = path_obj.restricted_penalty(None)

    # take a guess at the inverse step size
    if initial_step is None:
        _lipschitz = power_L(path_obj.X, max_its=50)
        path_obj.final_step = 1000. / _lipschitz 
    else:
        path_obj.final_step = initial_step

    objective = []
    solutions = []
    
    for lagrange in lagrange_seq:
        enet_loss = path_obj.enet_loss(lagrange)
        total_loss = smooth_sum([loss, enet_loss])
        penalty.lagrange = lagrange * path_obj.alpha

        problem = simple_problem(total_loss, penalty)
        problem.coefs[:] = solution
        solve_args['tol'] = inner_tol
        solve_args['start_step'] = path_obj.final_step
        solution = problem.solve(**solve_args)

        objective.append(loss.smooth_objective(solution, mode='func'))
        solutions.append(solution.T.copy())
        grad_solution = total_loss.smooth_objective(solution, 'grad')
        path_obj.final_step = problem.final_step

        if verbose:
            print({'lagrange':lagrange,
                   'sparsity':(solution != 0).sum(),
                   'dev explained':1. - objective[-1] / objective[0],
                   'point on path':list(lagrange_seq).index(lagrange),
                   'penalty':path_obj.penalty.seminorm(solution, lagrange=1),
                   'dual penalty':path_obj.penalty.conjugate.seminorm(grad_solution, lagrange=1)})

    objective = np.array(objective)
    output = {'devratio': 1 - objective / objective.max(),
              'lagrange': lagrange_seq,
              'beta':np.array(solutions)}

    return output


