import numpy as np
from scipy import sparse
import warnings
import inspect

from ..problems.composite import smooth as smooth_composite
from ..affine import affine_transform, linear_transform
from ..identity_quadratic import identity_quadratic

#TODO: create proximal methods for known smooth things
class smooth_atom(smooth_composite):

    """
    A class for representing a smooth function and its gradient
    """

    objective_template = r'''f(%(var)s)'''
    objective_vars = {'var':r'\beta',
                      'shape':'p',
                      'coef':'C',
                      'offset':r'\alpha+'}

    def __init__(self, shape, coef=1, offset=None,
                 quadratic=None, initial=None):
        smooth_composite.__init__(self, shape,
                                  offset=offset,
                                  quadratic=quadratic,
                                  initial=initial)
        self.coef = coef
        if coef < 0:
            raise ValueError('coefs must be nonnegative to ensure convexity (assuming all atoms are indeed convex)')
        self.coefs = np.zeros(self.shape)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        raise NotImplementedError
    
    @classmethod
    def affine(cls, linear_operator, offset, coef=1, diag=False,
               quadratic=None, **kws):
        """
        Keywords given in kws are passed to cls constructor along with other arguments
        """
        if not isinstance(linear_operator, affine_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)
        
        # the minus signs below for offset is there until affine transforms SUBTRACT 
        # their offset until add. 
        # for atoms, the offset is really the "center"

        atom = cls(l.output_shape, coef=coef, offset=-offset, quadratic=quadratic, **kws)
        
        return affine_smooth(atom, l)

    @classmethod
    def linear(cls, linear_operator, coef=1, diag=False,
               offset=None, 
               quadratic=None, **kws):
        """
        Keywords given in kws are passed to cls constructor along with other arguments
        """
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)

        atransform = affine_transform(linear_operator, None, diag=diag)
        atom = cls(atransform.output_shape, coef=coef, quadratic=quadratic, offset=offset, **kws)
        
        return affine_smooth(atom, atransform)

    @classmethod
    def shift(cls, offset, coef=1, quadratic=None, **kws):
        """
        Keywords given in kws are passed to cls constructor along with other arguments
        """
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)
        
        atom = cls(offset.shape, coef=coef, quadratic=quadratic, 
                   offset=offset, **kws)
        return atom

    def scale(self, obj, copy=False):
        if self.coef != 1:
            return obj * self.coef
        if copy:
            return obj.copy()
        return obj

    def get_conjugate(self):
        raise NotImplementedError('each smooth loss should implement its own get_conjugate')

    @property
    def conjugate(self):
        return self.get_conjugate()
 

def acceptable_init_args(cls, proposed_keywords):
    """
    Check that the keywords in the dictionary proposed_keywords are arguments to __init__ of class cls

    Returns True/False
    """
    args = inspect.getargspec(cls.__init__).args
    forbidden = ['self', 'shape', 'coef', 'quadratic', 'initial', 'offset']
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
 
    
    objective_vars = {'linear':'X'}

    def __init__(self, smooth_atom, atransform, store_grad=True, diag=False):
        self.store_grad = store_grad
        self.atom = smooth_atom
        if not isinstance(atransform, affine_transform):
            atransform = linear_transform(atransform, diag=diag)
        self.affine_transform = atransform
        self.shape = atransform.input_shape
        self.coefs = np.zeros(self.shape)

    def _get_coef(self):
        return self.atom.coef

    def _set_coef(self, coef):
        self.atom.coef = coef
    coef = property(_get_coef, _set_coef)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        eta = self.affine_transform.affine_map(x)
        if mode == 'both':
            v, g = self.atom.smooth_objective(eta, mode='both')
            if self.store_grad:
                self.grad = g
            g = self.affine_transform.adjoint_map(g).reshape(self.shape)
            return v, g
        elif mode == 'grad':
            g = self.atom.smooth_objective(eta, mode='grad')
            if self.store_grad:
                self.grad = g
            g = self.affine_transform.adjoint_map(g).reshape(self.shape)
            return g 
        elif mode == 'func':
            v = self.atom.smooth_objective(eta, mode='func')
            return v 

    @property
    def dual(self):
        try: 
            conj = self.atom.conjugate
            return self.affine_transform, conj
        except:
            return None

    def __repr__(self):
        return ("affine_smooth(%s, %s, store_grad=%s)" % 
                (str(self.atom),
                str(self.affine_transform),
                self.store_grad))

    def latexify(self, var=None, idx=''):
        template_dict = self.atom.objective_vars.copy()
        template_dict['linear'] = self.objective_vars['linear']
        if var is not None:
            template_dict['var'] = var
        template_dict['idx'] = idx
        return self.atom.latexify(var='%(linear)s_{%(idx)s}%(var)s' % template_dict, idx=idx)

class zero(smooth_atom):

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        if mode == 'both':
            return 0., np.zeros(x.shape)
        elif mode == 'func':
            return 0.
        elif mode == 'grad':
            return np.zeros(x.shape)
        raise ValueError("Mode not specified correctly")

class logistic_deviance(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    objective_template = r"""\ell^{L}\left(%(var)s\right)"""
    #TODO: Make init more standard, replace np.dot with shape friendly alternatives in case successes.shape is (n,1)

    def __init__(self, shape, successes, 
                 trials=None, coef=1., offset=None,
                 quadratic=None,
                 initial=None):

        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        if sparse.issparse(successes):
            #Convert sparse success vector to an array
            self.successes = successes.toarray().flatten()
        else:
            self.successes = np.asarray(successes)

        if trials is None:
            if not set([0,1]).issuperset(np.unique(self.successes)):
                raise ValueError("Number of successes is not binary - must specify number of trials")
            self.trials = np.ones(self.successes.shape, np.float)
        else:
            if np.min(trials-self.successes) < 0:
                raise ValueError("Number of successes greater than number of trials")
            if np.min(self.successes) < 0:
                raise ValueError("Response coded as negative number - should be non-negative number of successes")
            self.trials = trials * 1.

        saturated = self.successes / self.trials
        deviance_terms = np.log(saturated) * self.successes + np.log(1-saturated) * (self.trials - self.successes)
        deviance_constant = -2 * coef * deviance_terms[~np.isnan(deviance_terms)].sum()

        devq = identity_quadratic(0,0,0,-deviance_constant)
        self.quadratic += devq

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
                
            f, g = -2 * self.scale((np.dot(self.successes,x) - np.sum(self.trials*log_exp_x))), -2 * self.scale(self.successes-ratio)
            return f, g
        elif mode == 'grad':
            ratio = self.trials * 1.
            if overflow:
                ratio[not_overflow_ind] *= exp_x/(1.+exp_x)
            else:
                ratio *= exp_x/(1.+exp_x)
            f, g = None, - 2 * self.scale(self.successes-ratio)
            return g
        elif mode == 'func':
            if overflow:
                log_exp_x = x * 1.
                log_exp_x[not_overflow_ind] = np.log(1.+exp_x)
            else:
                log_exp_x = np.log(1.+exp_x)
            f, g = -2 * self.scale(np.dot(self.successes,x) - np.sum(self.trials * log_exp_x)), None
            return f
        else:
            raise ValueError("mode incorrectly specified")


class poisson_deviance(smooth_atom):

    """
    A class for combining the Poisson log-likelihood with a general seminorm
    """

    objective_template = r"""\ell^{P}\left(%(var)s\right)"""

    def __init__(self, shape, counts, coef=1., offset=None,
                 quadratic=None,
                 initial=None):

        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        if sparse.issparse(counts):
            #Convert sparse success vector to an array
            self.counts = counts.toarray().flatten()
        else:
            self.counts = counts

        if not np.allclose(np.round(self.counts),self.counts):
            raise ValueError("Counts vector is not integer valued")
        if np.min(self.counts) < 0:
            raise ValueError("Counts vector is not non-negative")

        saturated = counts
        deviance_terms = -2 * coef * ((counts - 1) * np.log(counts))
        deviance_terms[counts == 0] = 0

        deviance_constant = -2 * coef * deviance_terms.sum()

        devq = identity_quadratic(0,0,0,-deviance_constant)
        self.quadratic += devq

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
            f, g = -2. * self.scale(-np.sum(exp_x) + np.dot(self.counts,x)), -2. * self.scale(self.counts - exp_x)
            return f, g
        elif mode == 'grad':
            f, g = None, -2. * self.scale(self.counts - exp_x)
            return g
        elif mode == 'func':
            f, g =  -2. * self.scale(-np.sum(exp_x) + np.dot(self.counts,x)), None
            return f
        else:
            raise ValueError("mode incorrectly specified")


class multinomial_deviance(smooth_atom):

    """
    A class for baseline-category logistic regression for nominal responses (e.g. Agresti, pg 267)
    """

    objective_template = r"""\ell^{M}\left(%(var)s\right)"""

    def __init__(self, shape, counts, coef=1., offset=None,
                 initial=None,
                 quadratic=None):

        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

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

        if shape[1] != self.J - 1:
            raise ValueError("Primal shape is incorrect - should only have coefficients for first J-1 categories")

        saturated = self.counts / (1. * self.trials[:,np.newaxis])
        deviance_terms = np.log(saturated) * self.counts
        deviance_terms[np.isnan(deviance_terms)] = 0
        deviance_constant = -2 * coef * deviance_terms.sum()

        devq = identity_quadratic(0,0,0,-deviance_constant)
        self.quadratic += devq

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
        #JT: should be able to do this with np.newaxis

        if mode == 'both':
            ratio = ((self.trials/(1. + np.sum(exp_x, axis=1))) * exp_x.T).T
            f, g = -2. * self.scale(np.sum(self.firstcounts * x) -  np.dot(self.trials, np.log(1. + np.sum(exp_x, axis=1)))), - 2 * self.scale(self.firstcounts - ratio) 
            return f, g
        elif mode == 'grad':
            ratio = ((self.trials/(1. + np.sum(exp_x, axis=1))) * exp_x.T).T
            f, g = None, - 2 * self.scale(self.firstcounts - ratio) 
            return g
        elif mode == 'func':
            f, g = -2. * self.scale(np.sum(self.firstcounts * x) -  np.dot(self.trials, np.log(1. + np.sum(exp_x, axis=1)))), None
            return f
        else:
            raise ValueError("mode incorrectly specified")


def logistic_loss(X, Y, trials=None, coef=1.):
    '''
    Construct a logistic loss function for successes Y and
    affine transform X.

    Parameters
    ----------

    X : [affine_transform, ndarray]
        Design matrix

    Y : ndarray

    '''
    n = Y.shape[0]
    loss = affine_smooth(logistic_deviance(Y.shape, 
                                           Y,
                                           coef=coef/n,
                                           trials=trials), 
                         X)
    return loss

class sum(smooth_atom):
    """
    A simple way to combine smooth objectives
    """
    def __init__(self, atoms, weights=None):
        self.offset = None
        self.atoms = atoms
        if weights is None:
            weights = np.ones(len(self.atoms))
        self.weights = np.asarray(weights).reshape(-1)
        if self.weights.shape[0] != len(atoms):
            raise ValueError('weights and atoms have different lengths')
        self.coefs = self.atoms[0].coefs

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        x = self.apply_offset(x)
        f, g = 0, 0
        for w, atom in zip(self.weights, self.atoms):
            if mode == 'func':
                f += w * atom.smooth_objective(x, 'func')
            elif mode == 'grad':
                g += w * atom.smooth_objective(x, 'grad')
            elif mode == 'both':
                fa, ga = atom.smooth_objective(x, 'both')
                f += fa; g += ga

        if mode == 'func':
            return f
        elif mode == 'grad':
            return g
        elif mode == 'both':
            return f, g
        else:
            raise ValueError("mode incorrectly specified")
