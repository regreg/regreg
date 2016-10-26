from __future__ import print_function, division, absolute_import

import numpy as np
import warnings

from .identity_quadratic import identity_quadratic as sq

class algorithm(object):

    perform_backtrack = True
    step = None
    alpha = 1.1
    debug = False
    max_its=10000
    min_its=5
    default_tol=1e-5
    attempt_decrease = False # start trying to take bigger steps right away

    def __init__(self, composite):
        self.composite = composite

    @property
    def output(self):
        """
        Return the 'interesting' part of the composite arguments.
        In the regression case, this is the tuple (beta, r).
        """
        return self.composite.output

    def fit(self):
        """
        Abstract method.
        """
        raise NotImplementedError


class FISTA(algorithm):

    """
    The FISTA generalized gradient algorithm
    """

    alpha = 1.1 # multiplicative factor for changing step

    def fit(self,
            tol=None,
            min_its=None,
            max_its=None,
            FISTA=True,
            start_step=1.,
            restart=np.inf,
            coef_stop=False,
            return_objective_hist = True,
            monotonicity_restart = True,
            debug = None,
            prox_control={}):

        """
        Use the FISTA (or ISTA) algorithm to fit the problem

        Parameters
        ----------
        FISTA : bool
              use Nesterov weights? If False, this is just gradient descent
        start_step : float
              used in backtracking. This is the starting value of self.step
        restart : int
              Restart Nesterov weights every restart iterations. Default is never (np.inf)
        coef_stop : bool
              Stop based on coefficient changes instead of objective value
        return_objective_hist : bool
              Return the sequence of objective values?
        monotonicity_restart : bool
              If True, Nesterov weights are restarted every time the objective value increases
        debug : bool
              Resets self.debug, which controls whether convergence information is printed
        prox_control : dict
              A dictionary of arguments for fit(), used when the composite.proximal_step itself is a FISTA problem
    
        Returns
        -------

        objective_hist : ndarray
              A vector of objective values. Only return if return_objective_hist is True.

        """

        tol = tol or self.default_tol
        min_its = min_its or self.min_its
        max_its = max_its or self.max_its

        if debug is not None:
            self.debug = debug
        self.prox_control = {}

        objective_hist = np.zeros(max_its)
        
        if self.perform_backtrack and self.step is None:
            #If self.step is not available from last fit use start_step
            self.step = start_step

        self.working_coefs = self.composite.coefs

        # initialize the Nesterov weights
        self.weight_old = 1.

        working_smooth = self.composite.smooth_objective(self.working_coefs, mode='func')
        working_obj = working_smooth + self.composite.nonsmooth_objective(self.working_coefs, check_feasibility=True)
        
        if np.isnan(working_obj):
            raise ValueError('objective is NaN')

        itercount = 0
        badstep = 0
        while itercount < max_its:

            #Restart every 'restart' iterations

            if np.mod(itercount+1,restart)==0:
                if self.debug:
                    print("\tRestarting weights")
                self.working_coefs = self.composite.coefs
                t_old = 1.

            objective_hist[itercount] = working_obj

            # Backtracking loop
            if self.perform_backtrack:
                proposed_coefs, proposed_smooth = self.backtrack(itercount)
                     
            else:
                #Use specified Lipschitz constant
                working_grad = self.composite.smooth_objective(self.working_coefs, mode='grad')
                lipschitz = self.composite.lipschitz
                self.step = 1. / lipschitz
                if self.prox_control != {}:
                    proposed_coefs = self.composite.proximal_step(sq(lipschitz, self.working_coefs, working_grad, 0), prox_control=prox_control)
                else:
                    proposed_coefs = self.composite.proximal_step(sq(lipschitz, self.working_coefs, working_grad, 0))
                proposed_smooth = self.composite.smooth_objective(proposed_coefs, mode='func')

            proposed_obj = proposed_smooth + self.composite.nonsmooth_objective(proposed_coefs)

            obj_change = np.fabs(proposed_obj - working_obj)
            obj_rel_change = obj_change/np.max([np.fabs(working_obj),1.])
            if coef_stop:
                coef_rel_change = np.linalg.norm(self.composite.coefs - proposed_coefs) / np.max([1.,np.linalg.norm(proposed_coefs)])

            if self.debug:
                if coef_stop:
                    print(itercount, working_obj,
                          self.step, obj_rel_change,
                          coef_rel_change, tol)
                else:
                    print("%i    obj: %.6e    step: %.2e    "
                          "rel_obj_change: %.2e    tol: %.1e" %
                          (itercount, working_obj,
                           self.step, obj_rel_change,
                           tol))

            if itercount >= min_its:
                if coef_stop:
                    if coef_rel_change < tol:
                        self.working_coefs = proposed_coefs
                        if self.debug:
                            print("Success: Optimization stopped because "
                                  "change in coefficients was below tolerance")
                        break
                else:
                    if obj_rel_change < tol or obj_change < tol:
                        self.working_coefs = proposed_coefs
                        if self.debug:
                            print('Success: Optimization stopped because '
                                  'decrease in objective was below tolerance')
                        break

            self.update_working_coefs(proposed_coefs)

            if itercount > 1 and working_obj < proposed_obj and obj_rel_change > 1e-10 and monotonicity_restart:
                #Adaptive restarting: restart if monotonicity violated
                if self.debug:
                    print("%i Restarting weights" % itercount)
                self.attempt_decrease = True

                if self.weight_old == 1.:
                    #Gradient step didn't decrease objective: tolerance composites or incorrect prox op... time to give up?
                    if self.debug:
                        print("%i  Badstep: current: %f, proposed %f" %
                              (itercount, working_obj, proposed_obj))
                    badstep += 1
                    if badstep > 3:
                        warnings.warn('prox is taking bad steps')
                        if self.debug:
                            print('Caution: Optimization stopped while prox '
                                  'was taking bad steps')
                        break
                itercount += 1
                self.weight_old = 1.
                self.working_coefs = self.composite.coefs

            else:
                self.composite.coefs = proposed_coefs # XXX how much time do we waste on a copy here?
                self.weight_old = self.weight_new
                itercount += 1
                working_obj = proposed_obj

        if self.debug:
            if itercount == max_its:
                print("Optimization stopped because iteration limit was "
                      "reached")
            print("FISTA used", itercount, "of", max_its, "iterations")
        if return_objective_hist:
            return objective_hist[:itercount]

    def update_working_coefs(self, proposed_coefs):
        if FISTA:
            #Use Nesterov weights
            self.weight_new = 0.5 * (1 + np.sqrt(1+4*(self.weight_old**2)))
            #XXX should that be self.working_coefs or self.composite.coefs at the end of this line?
            self.working_coefs = proposed_coefs + ((self.weight_old-1)/(self.weight_new)) * (proposed_coefs - self.composite.coefs)
        else:
            #Just do ISTA
            self.weight_new = 1.
            self.working_coefs = proposed_coefs

    def backtrack(self, itercount):

        if np.mod(itercount+1,100)==0 or self.attempt_decrease:
            self.step *= self.alpha
            self.attempt_decrease = True
        working_smooth, working_grad = self.composite.smooth_objective(self.working_coefs, mode='both')
        while True:
            if self.prox_control != {}:
                proposed_coefs = self.composite.proximal_step(sq(1. / self.step, self.working_coefs, working_grad, 0), prox_control=prox_control)
            else:
                proposed_coefs = self.composite.proximal_step(sq(1. / self.step, self.working_coefs, working_grad, 0))

            proposed_smooth = self.composite.smooth_objective(proposed_coefs, mode='func')

            if not np.isfinite(proposed_smooth):
                stop = False
            elif np.fabs(proposed_smooth - working_smooth)/np.max([1., proposed_smooth]) > 1e-10:
                stop = (proposed_smooth <= working_smooth + np.dot((proposed_coefs - self.working_coefs).reshape(-1),
                                                                   working_grad.reshape(-1)) + \
                            (0.5/self.step)*np.linalg.norm(proposed_coefs-self.working_coefs)**2)
            else:
                proposed_grad = self.composite.smooth_objective(proposed_coefs, mode='grad')
                stop = (np.fabs(np.dot((proposed_coefs - self.working_coefs).reshape(-1),
                                       (working_grad - proposed_grad).reshape(-1))) <= 
                        (0.5/self.step)*np.linalg.norm(proposed_coefs-self.working_coefs)**2)
            if stop:
                break

            self.attempt_decrease = False
            self.step /= self.alpha
            if not self.step > 0:
                raise ValueError("stepsize zero reached")
            if self.debug:
                print("%i    Decreasing step to" % itercount, self.step)
        return proposed_coefs, proposed_smooth
