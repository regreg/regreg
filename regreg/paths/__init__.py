import gc
from warnings import warn
import numpy as np

from ..affine import astransform, power_L
from ..smooth import affine_smooth, sum as smooth_sum
from ..problems.simple import simple_problem

def subsample_columns(X, columns):

    X = astransform(X)
    cols = np.zeros((len(columns),) + X.output_shape)
    indicator = np.zeros(X.input_shape[0]) # assuming 1-dimensional input and output shape here
    for i, col in enumerate(columns):
        indicator[col] = 1 # 1-hot vector
        cols[i] = X.dot(indicator)
        indicator[col] = 0 # back to 0 vector
    return cols.T

class grouped_path(object):

    BIG = 1e12 # lagrange parameter for finding null solution

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
        Extract columns of X into ndarray or
        regreg transform
        """
        return subsample_columns(X, 
                                 columns)

    def full_gradient(self, 
                      saturated_loss, 
                      linear_predictor):
        """
        Gradient of saturated loss composed with self.X
        """
        saturated_grad = saturated_loss.smooth_objective(linear_predictor, 'grad')
        return self.X.T.dot(saturated_grad)

    def linpred(self,
                coef,
                X,
                test):
        """
        Form linear predictor for given set of coefficients.

        Typical use case will have coef of shape `(l,p)` for
        univariate response regressions and `(l,q,p)` for
        multiple response regressions (e.g. multinomial with
        `q` classes) where `l` is the number of Lagrange
        parameters.

        Parameters
        ----------

        coef : ndarray
            A set of coefficients. 

        X : object
            Representation of design matrix, usually `self.X` -- subclasses
            may not assume this is actually an array.

        test : index
            Indices of X.dot(coef.T) to return.

        Returns
        -------

        linpred : Matrix of linear predictors.

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
        raise NotImplementedError

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
             A weighted L1 penalty. If None, defaults
             to `self.penalty`.

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
                 max_tries=20):

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
    ever_active = path_obj.updated_ever_active([])

    dev_solution = path_obj.saturated_loss.smooth_objective(linear_predictor, 'func')
    solutions = [solution.T.copy()]
    dev_explained = [dev_solution]

    all_failing = np.zeros(path_obj.group_shape, np.bool)
    subproblem_set = path_obj.updated_ever_active([])

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
                                                  penalty=strong_penalty)

            if check_active:
                strong_failing = strong_A + (strong_I_ranks >= 0)
            else:
                strong_failing = strong_I_ranks >= 0

            if np.any(strong_failing):
                delta = np.zeros(path_obj.group_shape, np.bool)
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
                                                                path_obj.alpha * lagrange_new)

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
               solve_args={}):
    
    solution, _ = initial_data

    # basic setup

    loss = path_obj.full_loss()
    all_vars = np.ones(path_obj.penalty.shape, np.bool)
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


