import numpy as np
from scipy import sparse
import warnings
import inspect

from ..problems.composite import smooth as smooth_composite
from ..affine import affine_transform, linear_transform, astransform
from ..identity_quadratic import identity_quadratic

#TODO: create proximal methods for known smooth things
class smooth_atom(smooth_composite):

    """
    A class for representing a smooth function and its gradient

    Parameters
    ----------

    shape : tuple
       Shape of argument to `smooth_objective`

    coef : float (optional)
       Scalar multiple to be applied (must be nonnegative)

    offset : ndarray (optional)
       Vector to be subtracted before evaluating `smooth_objective`. 

    quadratic : `identity_quadratic` (optional)
       Instance of `identity_quadratic` to be added to overall
       objective.

    initial : ndarray (optional)
       Initial value for coefficients.

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

    def smooth_objective(self, arg, mode='both', check_feasibility=False):
        """

        Parameters
        ----------

        arg : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `beta` is not
            in the domain.

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `beta`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        raise NotImplementedError("Abstract method.")
    
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

    """

    Composition of a smooth objective with an affine transform.

    """

    force_reshape = True

    objective_vars = {'linear':'X'}

    def __init__(self, smooth_atom, atransform, store_grad=True, diag=False):
        """

        Parameters
        ----------

        smooth_atom : `regreg.smooth.smooth_atom`
             A smooth atom.

        atransform : `regreg.affine.affine_transform`
             An affine transformation, or cast to one
             using `regreg.affine.linear_transform`

        store_grad : bool
             If True, when computing the gradient,
             store a reference to the gradient of `smooth_atom`
             in the attribute `grad`.

        diag : bool
             Indicates if `atransform` acts diagonally,
             i.e. a rescaling.
             Passed to `regreg.affine.linear_transform`.

        """
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

    def smooth_objective(self, arg, mode='both', check_feasibility=False):
        """
        Compute the smooth objective at the point `self.transform.affine_map(arg)`.

        Parameters
        ----------

        arg : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `beta` is not
            in the domain.

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `self.transform(arg)`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """

        eta = self.affine_transform.affine_map(arg)
        if mode == 'both':
            v, g = self.atom.smooth_objective(eta, mode='both')
            if self.store_grad:
                self.grad = g
            g = self.affine_transform.adjoint_map(g)
            if self.force_reshape:
                g = g.reshape(self.shape)
            return v, g
        elif mode == 'grad':
            g = self.atom.smooth_objective(eta, mode='grad')
            if self.store_grad:
                self.grad = g
            g = self.affine_transform.adjoint_map(g)
            if self.force_reshape:
                g = g.reshape(self.shape)
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

        obj_latex = self.atom.latexify(var='%(linear)s_{%(idx)s}%(var)s' % template_dict, idx=idx)
        if not self.quadratic.iszero:
            return ' + '.join([obj_latex, self.quadratic.latexify(var=template_dict['var'], idx=idx)])
        else:
            return obj_latex

class zero(smooth_atom):

    """
    The zero function.
    """

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        if mode == 'both':
            return 0., np.zeros(x.shape)
        elif mode == 'func':
            return 0.
        elif mode == 'grad':
            return np.zeros(x.shape)
        raise ValueError("Mode not specified correctly")

class sum(smooth_atom):
    """
    A simple way to combine smooth objectives
    """
    def __init__(self, atoms, weights=None):
        """
        Parameters
        ----------

        atoms : sequence
            A sequence of `regreg.smooth.smooth_atom` that will be summed
            to make a new atom.

        weights : ndarray (optional)
            If provided, these weights will appear as coefficients
            in front of each atom.

        """
        self.offset = None
        self.atoms = atoms
        if weights is None:
            weights = np.ones(len(self.atoms))
        self.weights = np.asarray(weights).reshape(-1)
        if self.weights.shape[0] != len(atoms):
            raise ValueError('weights and atoms have different lengths')
        if np.any(self.weights < 0):
            raise ValueError('weights should be non-negative to maintain convexity')
        self.coefs = self.atoms[0].coefs
        self.shape = self.coefs.shape

    def smooth_objective(self, x, mode='both', check_feasibility=False):

        """
        Compute the smooth objective at the point `self.transform.affine_map(arg)`,
        which is the sum of each `atom`'s objective with its respective weight.

        Parameters
        ----------

        arg : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `beta` is not
            in the domain.

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `self.transform(arg)`, else if `mode` is 'grad' returns the gradient
        else returns both.
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

