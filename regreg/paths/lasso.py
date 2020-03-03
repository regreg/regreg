from __future__ import print_function, division, absolute_import

from warnings import warn
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
from ..atoms.weighted_atoms import l1norm as weighted_l1norm

class lasso_path(object):

    BIG = 1e12

    def __init__(self, 
                 saturated_loss,
                 X, 
                 lasso_weights,
                 elastic_net_param=None,
                 alpha=1.,  # elastic net mixing -- 1 is LASSO
                 lagrange_proportion=0.05,
                 nstep=100,
                 elastic_net_penalized=None):

        self.saturated_loss = saturated_loss
        self.X = astransform(X)

        # the penalty parameters

        self.alpha = alpha
        self.penalty = weighted_l1norm(lasso_weights, lagrange=1)
        self.shape = self.penalty.shape
        self.nstep = nstep

        # elastic net part
        if elastic_net_param is None:
            elastic_net_param = np.ones(self.shape)
        self.elastic_net_param = elastic_net_param

        # find lagrange_max

        unpenalized = self.penalty.weights == 0
        penalized = ~unpenalized
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
             null_linpred) = self.solve_subproblem(np.nonzero(unpenalized)[0],
                                                   self.BIG,
                                                   tol=1.e-8)
            self.linear_predictor = null_linpred
            self.solution[unpenalized] = null_soln
        else:
            self.linear_predictor = np.zeros(self.saturated_loss.shape)

        self.grad_solution = (self.full_gradient(self.saturated_loss, 
                                                 self.linear_predictor) + self.enet_grad(self.solution, 1))
        self.lagrange_max = self.get_lagrange_max(self.grad_solution) # penalty specific
        self.lagrange_sequence = self.lagrange_max * np.exp(np.linspace(np.log(lagrange_proportion), 
                                                                        0, 
                                                                        nstep))[::-1]

    def main(self, inner_tol=1.e-5, verbose=False):

        _lipschitz = power_L(self.X)

        # take a guess at the inverse step size
        self.final_step = 1000. / _lipschitz 
        lseq = self.lagrange_sequence # shorthand

        # gradient of restricted elastic net at lambda_max

        solution = self.solution
        grad_solution = self.grad_solution
        linear_predictor = self.linear_predictor
        ever_active = self.ever_active

        obj_solution = self.saturated_loss.smooth_objective(self.linear_predictor, 'func')
        solutions = []
        objective = [obj_solution]

        all_failing = np.zeros(grad_solution.shape, np.bool)
        subproblem_set = self.ever_active

        for lagrange_new, lagrange_cur in zip(lseq[1:], lseq[:-1]):
            tol = inner_tol
            num_tries = 0
            debug = False
            coef_stop = True

            while True:
                subproblem_set = subproblem_set + self.updated_ever_active(all_failing)

                (self.final_step, 
                 subproblem_grad, 
                 subproblem_soln,
                 subproblem_linpred) = self.solve_subproblem(np.nonzero(subproblem_set)[0],
                                                             lagrange_new,
                                                             tol=tol,
                                                             start_step=self.final_step,
                                                             debug=debug and verbose,
                                                             coef_stop=coef_stop)

                saturated_grad = self.saturated_loss.smooth_objective(subproblem_linpred, 'grad')
                # as subproblem always contains ever active, 
                # rest of solution should be 0
                solution[subproblem_set] = subproblem_soln

                # strong rules step

                strong = self.strong_set(lagrange_cur * self.alpha, 
                                         lagrange_new * self.alpha, 
                                         grad_solution)
                strong_enet_grad = self.enet_grad(solution,
                                                  lagrange_new,
                                                  subset=strong)
                strong_soln = solution[strong]
                X_strong = self.subsample_columns(self.X, np.nonzero(strong)[0])
                strong_grad = (X_strong.T.dot(saturated_grad) +
                               strong_enet_grad)
                strong_failing = self.check_KKT(strong_grad, strong_soln, self.alpha * lagrange_new, subset=strong) 

                if np.any(strong_failing):
                    delta = np.zeros(self.shape, np.bool)
                    delta[strong] = strong_failing
                    all_failing += delta 
                else:
                    enet_grad = self.enet_grad(solution, 
                                               lagrange_new)
                    grad_solution[:] = (self.full_gradient(self.saturated_loss, 
                                                           subproblem_linpred) + 
                                        enet_grad)
                    all_failing = self.check_KKT(grad_solution, solution, self.alpha * lagrange_new)

                    if not all_failing.sum():
                        self.ever_active[:] = self.updated_ever_active(solution != 0)
                        linear_predictor[:] = subproblem_linpred
                        break
                    else:
                        if verbose:
                            print('failing:', np.nonzero(all_failing)[0])
                num_tries += 1

                tol /= 2.
                if num_tries % 5 == 0:

                    solution[subproblem_set] = subproblem_soln
                    solution[~subproblem_set] = 0

                    enet_grad = self.enet_grad(solution, 
                                               lagrange_new)
                    grad_solution[:] = (self.full_gradient(self.saturated_loss, 
                                                           subproblem_linpred) + 
                                        enet_grad)
                    debug = True
                    tol = inner_tol
                    if num_tries >= 20:
                        warn('convergence not achieved for lagrange=%0.4e' % lagrange_new)
                        break

                subproblem_set += all_failing

            self.ever_active[:] = ever_active
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

    # method potentially overwritten in subclasses for penalty considerations

    def get_lagrange_max(self,
                         grad_solution):
        penalized = self.penalty.weights > 0
        return np.fabs(grad_solution[penalized] / self.penalty.weights[penalized]).max()

    def check_KKT(self,
                  grad_solution,
                  solution,
                  lagrange_new,
                  subset=None):

        if subset is not None:
            weights = self.penalty.weights[subset]
        else:
            weights = self.penalty.weights

        value = _check_KKT(weights, 
                           grad_solution, 
                           solution, 
                           lagrange_new)
        return value > 0

    def strong_set(self,
                   lagrange_cur,
                   lagrange_new,
                   grad_solution):

        return _strong_set(self.penalty.weights,
                           lagrange_cur,
                           lagrange_new,
                           grad_solution)

    def solve_subproblem(self, candidate_set, lagrange_new, **solve_args):
    
        # solve a problem with a candidate set

        sub_loss, sub_penalty, sub_X = _restricted_problem(self.X, 
                                                           self.saturated_loss, 
                                                           self.alpha, 
                                                           lagrange_new * self.penalty.weights, 
                                                           candidate_set,
                                                           self.subsample_columns)
        if self.alpha < 1:
            sub_elastic_net = _restricted_elastic_net(self.elastic_net_param, 
                                                      self.penalty.weights,
                                                      lagrange_new,
                                                      self.alpha,
                                                      candidate_set)
            sub_loss = smooth_sum([sub_loss, sub_elastic_net])

        sub_problem = simple_problem(sub_loss, sub_penalty)
        sub_problem.coefs[:] = self.solution[candidate_set] # warm start
        sub_soln = sub_problem.solve(**solve_args)
        sub_grad = sub_loss.smooth_objective(sub_soln, mode='grad') 
        sub_linear_pred = sub_X.dot(sub_soln)
        return sub_problem.final_step, sub_grad, sub_soln, sub_linear_pred

    def enet_grad(self,
                  solution,
                  lagrange_new,
                  subset=None):

        weights = self.penalty.weights
        elastic_net_param = self.elastic_net_param
        if subset is not None:
            solution = solution[subset]
            weights = weights[subset]
            elastic_net_param = elastic_net_param[subset]
        G = (1 - self.alpha) * solution * elastic_net_param
        G[weights != 0] *= lagrange_new
        return G

    def updated_ever_active(self,
                            bool_info):
        bool_info.reshape(self.shape)
        return self.ever_active + bool_info

    # Some common loss factories

    @classmethod
    def logistic(cls, X, Y, *args, **keyword_args):
        Y = np.asarray(Y)
        return cls(glm.logistic_loglike(Y.shape, Y), X, *args, **keyword_args)

    @classmethod
    def gaussian(cls, X, Y, *args, **keyword_args):
        Y = np.asarray(Y)
        return cls(glm.gaussian_loglike(Y.shape, Y), X, *args, **keyword_args)

# private functions

def _strong_set(lasso_weights,
                lagrange_cur,
                lagrange_new,
                gradient):

    scaled_grad = np.fabs(gradient)
    scaled_grad[lasso_weights > 0] /= lasso_weights[lasso_weights > 0]
    scaled_grad[lasso_weights == 0] = np.inf
    return scaled_grad > 2 * lagrange_new - lagrange_cur

def _restricted_elastic_net(elastic_net_params, 
                            lasso_weights, 
                            lagrange, 
                            alpha, 
                            candidate_set):
    penalized = lasso_weights != 0
    new_params = elastic_net_params * (1 - alpha)
    new_params[penalized] *= lagrange 
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
    Check whether (grad, solution) satisfy
    KKT conditions at a given tolerance.
    """

    INACTIVE = 2
    ACTIVE = 1
    UNPENALIZED = 3
    active = (solution != 0) * lasso_weights > 0
    failing = np.zeros(solution.shape)
    failing[active] = (np.fabs(grad[active] / (lagrange * lasso_weights[active]) + np.sign(solution[active])) > tol) * ACTIVE
    inactive = (solution == 0) * lasso_weights > 0
    failing[inactive] = (np.fabs(grad[inactive] / (lagrange * lasso_weights[inactive])) > 1 + tol) * INACTIVE
    unpenalized = lasso_weights == 0
    failing[unpenalized] = (np.fabs(grad[unpenalized] / np.mean(lasso_weights[lasso_weights>0])) > tol) * UNPENALIZED

    return failing 
