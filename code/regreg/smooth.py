import numpy as np
from scipy import sparse
from affine import affine_transform, linear_transform
import warnings
import inspect
from composite import smooth as smooth_composite

class smooth_atom(smooth_composite):

    """
    A class for representing a smooth function and its gradient
    """

    def __init__(self, primal_shape, coef=1, constant_term=0,
                 linear_term=None, offset=None):


        self.constant_term = constant_term
        if offset is not None:
            self.offset = np.array(offset)

        self.linear_term = None
        if linear_term is not None:
            self.linear_term = np.array(linear_term)

        self.primal_shape = primal_shape
        self.coef = coef
        if coef < 0:
            raise ValueError('coefs must be nonnegative to ensure convexity (assuming all atoms are indeed convex)')
        self.coefs = np.zeros(self.primal_shape)
        raise NotImplementedError

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        raise NotImplementedError
    
    @classmethod
    def affine(cls, linear_operator, offset, coef=1, diag=False,
               linear_term=None, 
               constant_term=0, **kws):
        """
        Keywords given in kws are passed to cls constructor along with other arguments
        """
        if not isinstance(linear_operator, affine_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)
        
        atom = cls(l.primal_shape, coef=coef, constant_term=constant_term, offset=offset, linear_term=linear_term, **kws)
        
        return affine_smooth(atom, l)

    @classmethod
    def linear(cls, linear_operator, coef=1, diag=False,
               offset=None, 
               constant_term=0, linear_term=None, **kws):
        """
        Keywords given in kws are passed to cls constructor along with other arguments
        """
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)

        atransform = affine_transform(linear_operator, None, diag=diag)
        atom = cls(atransform.primal_shape, coef=coef, constant_term=constant_term, linear_term=linear_term, offset=offset, **kws)
        
        return affine_smooth(atom, atransform)

    @classmethod
    def shift(cls, offset, coef=1, 
              constant_term=0, linear_term=None, **kws):
        """
        Keywords given in kws are passed to cls constructor along with other arguments
        """
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)
        
        atom = cls(offset.shape, coef=coef, constant_term=constant_term, offset=offset, linear_term=linear_term, **kws)
        return atom

    def scale(self, obj, copy=False):
        if self.coef != 1:
            return obj * self.coef
        if copy:
            return obj.copy()
        return obj

    def apply_offset(self, x):
        if self.offset is not None:
            return x + self.offset
        return x

    def apply_linear_term(self, f, g, mode='both'):
        if self.linear_term is not None:
            if mode == 'both':
                f += np.sum(x * self.linear_term)
                g += self.linear_term
                return f, g
            elif mode == 'func':
                f += np.sum(x * self.linear_term)
                return f
            elif mode == 'grad':
                g += self.linear_term
                return g
        else:
            if mode == 'both':
                return f, g
            elif mode == 'func':
                return f
            elif mode == 'grad':
                return g

    def get_conjugate(self, epsilon=0):
        # multiple of identity quadratic / 2 added 
        # before computing conjugate
        pass

def acceptable_init_args(cls, proposed_keywords):
    """
    Check that the keywords in the dictionary proposed_keywords are arguments to __init__ of class cls

    Returns True/False
    """
    args = inspect.getargspec(cls.__init__).args
    forbidden = ['self', 'primal_shape', 'coef', 'constant_term']
    for kw in proposed_keywords.keys():
        if not kw in args:
            return False
        if kw in forbidden:
            return False
    return True

class affine_smooth(smooth_atom):

    # if smooth_obj is a class, an object is created
    # smooth_obj(*args, **keywords)
    # else, it is assumed to be an instance of smooth_function
 
    def __init__(self, smooth_atom, atransform, store_grad=True):
        self.store_grad = store_grad
        self.sm_atom = smooth_atom
        self.affine_transform = atransform
        self.primal_shape = self.affine_transform.primal_shape
        self.coefs = np.zeros(self.primal_shape)

    def _get_coef(self):
        return self.sm_atom.coef

    def _set_coef(self, coef):
        self.sm_atom.coef = coef
    coef = property(_get_coef, _set_coef)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        eta = self.affine_transform.affine_map(x)
        if mode == 'both':
            v, g = self.sm_atom.smooth_objective(eta, mode='both')
            g = self.affine_transform.adjoint_map(g)
            if self.store_grad:
                self.grad = g
            return v, g
        elif mode == 'grad':
            g = self.sm_atom.smooth_objective(eta, mode='grad')
            g = self.affine_transform.adjoint_map(g)
            if self.store_grad:
                self.grad = g
            return g 
        elif mode == 'func':
            v = self.sm_atom.smooth_objective(eta, mode='func')
            return v 

def squaredloss(linear_operator, offset, coef=1):
    # the affine method gets rid of the need for the squaredloss class
    # as previously written squared loss had a factor of 2

    #return quadratic.affine(-linear_operator, offset, coef=coef/2., initial=np.zeros(linear_operator.shape[1]))
    return quadratic.affine(-linear_operator, offset, coef=coef/2.)

class quadratic(smooth_atom):
    """
    The square of the l2 norm
    """

    def __init__(self, primal_shape, coef=None, Q=None, Qdiag=False,
                 constant_term=0,
                 offset=None,
                 linear_term=None):
        self.offset = offset
        self.linear_term = linear_term
        self.constant_term = constant_term
        self.Q = Q
        if self.Q is not None:
            self.Q_transform = affine_transform(Q, 0, Qdiag)
        if type(primal_shape) == type(1):
            self.primal_shape = (primal_shape,)
        else:
            self.primal_shape = primal_shape
        self.coef = coef

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        x = self.apply_offset(x)
        if self.Q is None:
            if mode == 'both':
                f, g  = self.scale(np.linalg.norm(x)**2) + self.constant_term, self.scale(2 * x)
            elif mode == 'grad':
                f, g = None, self.scale(2 * x)
            elif mode == 'func':
                f, g = self.scale(np.linalg.norm(x)**2) + self.constant_term, None
            else:
                raise ValueError("mode incorrectly specified")
        else:
            if mode == 'both':
                f, g = self.scale(np.sum(x * self.Q_transform.linear_map(x))) + self.constant_term, self.scale(2 * self.Q_transform.linear_map(x))
            elif mode == 'grad':
                f, g = None, self.scale(2 * self.Q_transform.linear_map(x))
            elif mode == 'func':
                f, g = self.scale(np.sum(x * self.Q_transform.linear_map(x))) + self.constant_term, None
            else:
                raise ValueError("mode incorrectly specified")
        return self.apply_linear_term(f, g, mode)

class linear(smooth_atom):

    def __init__(self, vector, coef=1, linear_term=None,
                 offset=None, constant_term=0):
        self.linear_term = linear_term
        self.offset = offset
        self.vector = coef * vector
        self.primal_shape = vector.shape
        self.coef =1
        self.coefs = np.zeros(self.primal_shape)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        x = self.apply_offset(x)
        if mode == 'both':
            f, g = np.dot(self.vector, x), self.vector
        elif mode == 'grad':
            f, g = None, self.vector
        elif mode == 'func':
            f, g = np.dot(self.vector, x), None
        else:
            raise ValueError("mode incorrectly specified")
        return self.apply_linear_term(f, g, mode)

def signal_approximator(offset, coef=1):
    return quadratic.shift(-offset, coef)

class logistic_loglikelihood(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    #TODO: Make init more standard, replace np.dot with shape friendly alternatives in case successes.shape is (n,1)

    def __init__(self, primal_shape, successes, trials=None, coef=1., constant_term=0., linear_term=None, offset=None):

        self.linear_term = linear_term
        self.offset = offset
        self.constant_term = constant_term

        if sparse.issparse(successes):
            #Convert sparse success vector to an array
            self.successes = successes.toarray().flatten()
        else:
            self.successes = successes

        if trials is None:
            if not set([0,1]).issuperset(np.unique(self.successes)):
                raise ValueError("Number of successes is not binary - must specify number of trials")
            self.trials = np.ones(self.successes.shape)
        else:
            if np.min(trials-self.successes) < 0:
                raise ValueError("Number of successes greater than number of trials")
            if np.min(self.successes) < 0:
                raise ValueError("Response coded as negative number - should be non-negative number of successes")
            self.trials = trials

        self.constant_term = constant_term
        self.primal_shape = primal_shape
        self.coef = coef

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        
        #Check for overflow in np.exp (can occur during initial backtracking steps)
        x = self.apply_offset(x)
        if np.max(x) > 1e2:
            overflow = True
            not_overflow_ind = np.where(x <= 1e2)[0]
            exp_x = np.exp(x[not_overflow_ind])
        else:
            overflow = False
            exp_x = np.exp(x)
            
        if mode == 'both':
            ratio = self.trials * 1.
            if overflow:
                log_exp_x = x * 1.
                log_exp_x[not_overflow_ind] = np.log(1.+exp_x)
                ratio[not_overflow_ind] *= exp_x/(1.+exp_x)
            else:
                log_exp_x = np.log(1.+exp_x)
                ratio *= exp_x/(1.+exp_x)
                
            f, g = -2 * self.scale((np.dot(self.successes,x) - np.sum(self.trials*log_exp_x))) + self.constant_term, -2 * self.scale(self.successes-ratio)

        elif mode == 'grad':
            ratio = self.trials * 1.
            if overflow:
                ratio[not_overflow_ind] *= exp_x/(1.+exp_x)
            else:
                ratio *= exp_x/(1.+exp_x)
            f, g = None, - 2 * self.scale(self.successes-ratio)
        
        elif mode == 'func':
            if overflow:
                log_exp_x = x * 1.
                log_exp_x[not_overflow_ind] = np.log(1.+exp_x)
            else:
                log_exp_x = np.log(1.+exp_x)
            f, g = -2 * self.scale(np.dot(self.successes,x) - np.sum(self.trials * log_exp_x)) + self.constant_term, None
        else:
            raise ValueError("mode incorrectly specified")
        return self.apply_linear_term(f, g, mode)

class poisson_loglikelihood(smooth_atom):

    """
    A class for combining the Poisson log-likelihood with a general seminorm
    """

    #TODO: Make init more standard, replace np.dot with shape friendly alternatives in case successes.shape is (n,1)

    def __init__(self, primal_shape, counts, coef=1., constant_term=0.,
                 linear_term=None, offset=None):
        self.linear_term = linear_term
        self.offset = offset
        self.constant_term = constant_term
        
        if sparse.issparse(counts):
            #Convert sparse success vector to an array
            self.counts = counts.toarray().flatten()
        else:
            self.counts = counts

        if not np.allclose(np.round(self.counts),self.counts):
            raise ValueError("Counts vector is not integer valued")
        if np.min(self.counts) < 0:
            raise ValueError("Counts vector is not non-negative")
        

        self.primal_shape = primal_shape
        self.coef = coef

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        x = self.apply_offset(x)
        exp_x = np.exp(x)
        
        if mode == 'both':
            f, g = -2. * self.scale(-np.sum(exp_x) + np.dot(self.counts,x)) + self.constant_term, -2. * self.scale(self.counts - exp_x)
        elif mode == 'grad':
            f, g = None, -2. * self.scale(self.counts - exp_x)
        elif mode == 'func':
            f, g =  -2. * self.scale(-np.sum(exp_x) + np.dot(self.counts,x)) + self.constant_term, None
        else:
            raise ValueError("mode incorrectly specified")
        return self.apply_linear_term(f, g, mode)




class multinomial_loglikelihood(smooth_atom):

    """
    A class for baseline-category logistic regression for nominal responses (e.g. Agresti, pg 267)
    """

    def __init__(self, primal_shape, counts, coef=1., constant_term=0.,
                 linear_term=None, offset=None):

        self.linear_term = linear_term
        self.offset = offset
        self.constant_term = constant_term

        if sparse.issparse(counts):
            #Convert sparse success vector to an array
            self.counts = counts.toarray()
        else:
            self.counts = counts

        self.J = self.counts.shape[1]
        #Select the counts for the first J-1 categories
        self.firstcounts = self.counts[:,range(self.J-1)]

        if not np.allclose(np.round(self.counts),self.counts):
            raise ValueError("Counts vector is not integer valued")
        if np.min(self.counts) < 0:
            raise ValueError("Counts vector is not non-negative")

        self.trials = np.sum(self.counts, axis=1)

        if primal_shape[1] != self.J - 1:
            raise ValueError("Primal shape is incorrect - should only have coefficients for first J-1 categories")
        
        self.primal_shape = primal_shape
        self.coef = coef

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        x = self.apply_offset(x)
        exp_x = np.exp(x)

        #TODO: Using transposes to scale the rows of a 2d array - should we use an affine_transform to do this?

        if mode == 'both':
            ratio = ((self.trials/(1. + np.sum(exp_x, axis=1))) * exp_x.T).T
            f, g = -2. * self.scale(np.sum(self.firstcounts * x) -  np.dot(self.trials, np.log(1. + np.sum(exp_x, axis=1)))) + self.constant_term, - 2 * self.scale(self.firstcounts - ratio) 
        elif mode == 'grad':
            ratio = ((self.trials/(1. + np.sum(exp_x, axis=1))) * exp_x.T).T
            f, g = None, - 2 * self.scale(self.firstcounts - ratio) 
        elif mode == 'func':
            f, g = -2. * self.scale(np.sum(self.firstcounts * x) -  np.dot(self.trials, np.log(1. + np.sum(exp_x, axis=1)))) + self.constant_term, None
        else:
            raise ValueError("mode incorrectly specified")
        return self.apply_linear_term(f, g, mode)

