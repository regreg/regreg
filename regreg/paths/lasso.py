from __future__ import print_function, division, absolute_import

from warnings import warn
import gc

import numpy as np
import numpy.linalg as npl

import scipy.sparse

from ..affine import power_L, normalize, astransform
from ..atoms.seminorms import l1norm, constrained_positive_part
from ..smooth import sum as smooth_sum, affine_smooth
from ..problems.simple import simple_problem
from ..identity_quadratic import identity_quadratic as iq
from ..atoms.weighted_atoms import l1norm as weighted_l1norm

class lasso_path(object):

    def __init__(self, 
                 loss_factory, 
                 X, 
                 elastic_net=iq(0,0,0,0),
                 alpha=0.,  # elastic net mixing
                 lagrange_proportion=0.05,
                 nstep=100,
                 scale=True,
                 center=True):


        self.loss_factory = loss_factory

        self.scale = scale
        self.center = center

        # the penalty parameters
        self.alpha = alpha
        self.lagrange_proportion = lagrange_proportion
        self.nstep = nstep
        self._elastic_net = elastic_net.collapsed()

        self.X = astransform(X)
        self.ever_active = np.zeros(self.X.input_shape, np.bool)
        
    @property
    def shape(self):
        if self.scale or self.center:
            return self.Xn.output_shape[0], self.Xn.input_shape[0]
        else:
            return self.Xn.shape

    @property
    def nonzero(self):
        return self._selector

    @property
    def elastic_net(self):
        q = self._elastic_net
        q.coef *= self.lagrange
        q.linear_term *= self.lagrange
        return q

    @property
    def Xn(self):
        return self._Xn

    @property
    def loss(self):
        if not hasattr(self, '_loss'):
            self._loss = self.loss_factory(self._Xn)
        return self._loss

    @property
    def null_solution(self):
        if not hasattr(self, "_null_soln"):
            n, p = self.shape
            self._null_soln = np.zeros(p)
            null_problem, null_selector = self.restricted_problem(self.initial_active, self.lagrange_max)[:2]
            self._null_soln = null_selector.adjoint_map(null_problem.solve())
        return self._null_soln

    @property
    def lagrange_max(self):
        if not hasattr(self, "_lagrange_max"):
            null_soln = self.null_solution
            null_grad = self.loss.smooth_objective(null_soln, 'grad')
            self.penalty = mixed_lasso(self.penalty_structure, 1., weights=self.group_weights)
            conj = self.penalty.conjugate
            self._lagrange_max = conj.seminorm(null_grad)

        return self._lagrange_max

    def get_lagrange_sequence(self):
        if not hasattr(self, "_lagrange_sequence"):
            self._lagrange_sequence = self.lagrange_max * np.exp(np.linspace(np.log(self.lagrange_proportion), 0, 
                                                                             self.nstep))[::-1]
        return self._lagrange_sequence

    def set_lagrange_sequence(self, lagrange_sequence):
        self._lagrange_sequence = lagrange_sequence
    
    lagrange_sequence = property(get_lagrange_sequence, set_lagrange_sequence)

    @property
    def problem(self):
        p = self.shape[1]
        if not hasattr(self, "_problem"):
            self._problem = self.restricted_problem(np.ones(self.shape[1], np.bool),
                                                    self.lagrange_max)[0]
        return self._problem

    def get_lagrange(self):
        return self._problem.proximal_atom.lagrange

    def set_lagrange(self, lagrange):
        proximal_atom = self._problem.proximal_atom
        proximal_atom.lagrange = lagrange
        self.penalty.lagrange = lagrange
    lagrange = property(get_lagrange, set_lagrange)

    @property
    def solution(self):
        return self.problem.coefs

    @property
    def active(self):
        return self.solution != 0

    @property
    def lipschitz(self):
        if not hasattr(self, "_lipschitz"):
            self._lipschitz = power_L(self.Xn)
        return self._lipschitz

    def grad(self, loss=None):
        '''
        Gradient at current value. This includes the gradient
        of the smooth loss as well as the gradient of the elastic net part.
        This is used for determining whether the KKT conditions are met
        and which coefficients are in the strong set.
        '''
        if loss is None:
            loss = self.loss
        gsmooth = self.loss.smooth_objective(self.solution, 'grad')
        penalized = self.penalty_structure != UNPENALIZED
        # XXX the elastic net is probably not quite right here if the elastic net has a non-zero center
        gquad = self.elastic_net.objective(self.solution[penalized], 'grad')
        gsmooth[penalized] += gquad

        return gsmooth

    def strong_set(self, lagrange_cur, lagrange_new, grad=None,
                   slope_estimate=1):
        if grad is None:
            grad = self.grad()

        return strong_set_ml(self.penalty, lagrange_cur, lagrange_new, grad, slope_estimate)

    def slice_columns(self, columns):
        if self.scale or self.center:
            Xslice = self.Xn.slice_columns(columns)
        else:
            Xslice = self.Xn[:,columns]
        return Xslice

    def construct_loss(self, candidate_set, lagrange):
        Xslice = self.slice_columns(candidate_set)
        loss = self.loss_factory(Xslice)
        if self.intercept:
            Xslice.intercept_column = 0
        return Xslice, loss

    def restricted_problem(self, candidate_set, lagrange):
        '''
        Assumes the candidate set includes intercept as first column.
        '''

        Xslice, loss = self.construct_loss(candidate_set, lagrange)

        restricted_penalty_structure = self.penalty_structure[candidate_set]
        rps = restricted_penalty_structure # shorthand

        sliced_penalty = mixed_lasso(rps, lagrange, weights=self.group_weights)
        problem_sliced = simple_problem(loss, sliced_penalty)
        candidate_selector = selector(candidate_set, self.shape[1])
        return problem_sliced, candidate_selector, restricted_penalty_structure

    def solve_subproblem(self, candidate_set, lagrange_new, **solve_args):
    
        # try to solve the problem with the active set
        subproblem, selector, penalty_structure = self.restricted_problem(candidate_set, lagrange_new)
        subproblem.coefs[:] = selector.linear_map(self.solution)
        sub_soln = subproblem.solve(**solve_args)
        self.solution[:] = selector.adjoint_map(sub_soln)

        grad = subproblem.smooth_objective(sub_soln, mode='grad') 
        self.final_step = subproblem.final_step
        return self.final_step, grad, sub_soln, penalty_structure

    def main(self, inner_tol=1.e-5, verbose=False):

        # scaling will be needed to get coefficients on original scale   
        if self.scale:
            scalings = np.asarray(self.Xn.col_stds).reshape(-1)
        else:
            scalings = np.ones(self.shape[1])
        scalings = self.nonzero.adjoint_map(scalings)

        # take a guess at the inverse step size
        self.final_step = 1000. / self.lipschitz 
        lseq = self.lagrange_sequence # shorthand

        # first solution corresponding to all zeros except intercept 

        self.solution[:] = self.null_solution.copy()

        grad_solution = self.grad().copy()
        strong, strong_selector = self.strong_set(lseq[0], lseq[1], grad=grad_solution)

        p = self.shape[0]

        rescaled_solutions = scipy.sparse.csr_matrix(self.nonzero.adjoint_map(self.solution) 
                                                     / scalings)

        objective = [self.loss.smooth_objective(self.solution, 'func')]
        # not quite right -- should check tight constraints
        dfs = [np.sum(self.initial_active)]
        retry_counter = 0

        all_failing = np.zeros(grad_solution.shape, np.bool)

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

                self.solution[subproblem_set][:] = sub_soln
                # this only corrects the gradient on the subproblem_set
                grad_solution[subproblem_set][:] = grad

                strong_problem = self.restricted_problem(strong, lagrange_new)[0]
                strong_soln = self.solution[strong]
                strong_grad = (strong_problem.smooth_objective(strong_soln, mode='grad') + 
                               self.elastic_net[strong].objective(strong_soln, mode='grad'))
                strong_penalty = strong_problem.proximal_atom

                strong_failing = check_KKT(strong_penalty, strong_grad, strong_soln, lagrange_new) 

                if np.any(strong_failing):
                    all_failing += (
                        strong_selector.adjoint_map(strong_failing) != 0)
                else:
                    self.solution[subproblem_set][:] = sub_soln
                    grad_solution = self.grad()
                    all_failing = check_KKT(self.penalty, grad_solution, self.solution, lagrange_new)

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

                    self.solution[subproblem_set][:] = sub_soln
                    self.solution[~subproblem_set][:] = 0
                    grad_solution = self.grad()

                    debug = True
                    tol = inner_tol
                    if num_tries >= 10:
                        warn('convergence not achieved for lagrange=%0.4e' % lagrange_new)
                        break

            rescaled_solution = self.nonzero.adjoint_map(self.solution)
            rescaled_solutions = scipy.sparse.vstack([rescaled_solutions, rescaled_solution])
            objective.append(self.loss.smooth_objective(self.solution, mode='func'))
            dfs.append(self.ever_active.shape[0])
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
        return cls(logistic_factory(Y), X, *args, **keyword_args)

    @classmethod
    def squared_error(cls, X, Y, *args, **keyword_args):
        return cls(squared_error_factory(Y), X, *args, **keyword_args)

