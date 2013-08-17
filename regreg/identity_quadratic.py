r"""
This module contains a single class that is meant to represent
a quadratic of the form

.. math::

   \frac{L}{2} \|x-\mu\|^2_2 + \langle \eta, x \rangle + \gamma

with :math:`L, \mu, \eta, \gamma` = (coef, center, linear_term,
constant_term).
"""

from copy import copy

from numpy.linalg import norm
from numpy import all, asarray, allclose

class identity_quadratic(object):

    r"""
    This object is a quadratic function

    .. math::

        x \mapsto \frac{L}{2} \|x-\mu\|^2_2 + \langle \eta, x \rangle + \gamma

    used in the proximal methods of all atoms.

    """

    def __eq__(self, other):
        if isinstance(other, identity_quadratic):
            return (allclose(self.coef, other.coef) and
                    allclose(self.center, other.center) and
                    allclose(self.linear_term, other.linear_term) and
                    allclose(self.constant_term, other.constant_term))

    def __init__(self, coef, center, linear_term, constant_term=0):

        if center is not None:
            center = asarray(center)
            if center.shape == ():
                center = float(center)
        if linear_term is not None:
            linear_term = asarray(linear_term)
            if linear_term.shape == ():
                linear_term = float(linear_term)

        if coef is None:
            self.coef = 0.
        else:
            self.coef = coef

        self.center = center
        self.linear_term = linear_term
        if constant_term is None:
            self.constant_term = 0
        else:
            self.constant_term = constant_term

    @property
    def iszero(self):
        return all([self.coef in [0, None],
                    self.center is None or all(self.center == 0),
                    self.linear_term is None or all(self.linear_term == 0),
                    self.constant_term in [0, None]])

    def __copy__(self):
        return identity_quadratic(self.coef,
                                  copy(self.center),
                                  copy(self.linear_term),
                                  copy(self.constant_term))

    def noneify(self):
        '''
        replace zeros with nones
        '''
        if self.coef is None:
            self.coef = 0
        if self.constant_term is None:
            self.constant_term = 0
        if self.linear_term is not None and all(self.linear_term == 0):
            self.linear_term = None
        if self.center is not None and all(self.center_term == 0):
            self.center_term = None

    def zeroify(self):
        for a in ['coef', 'center', 'linear_term', 'constant_term']:
            if getattr(self, a) is None:
                setattr(self, a, 0)

    def recenter(self, offset):

        if offset is not None and all(offset == 0):
            offset = None

        if offset is not None:
            cpq = copy(self)
            cpq.center -= offset 
            cpq = cpq.collapsed()
            return offset, cpq
        else:
            return None, self.collapsed()


    def objective(self, x, mode='both'):
        coef, center, linear_term = self.coef, self.center, self.linear_term
        cons = self.constant_term
        if linear_term is None:
            linear_term = 0
        if center is not None:
            r = x - center
        else:
            r = x
        if mode == 'both':
            if linear_term is not None:
                return (norm(r)**2 * coef / 2. + (linear_term * x).sum() 
                        + cons, coef * r + linear_term)
            else:
                return (norm(r)**2 * coef / 2. + cons,
                        coef * r)
        elif mode == 'func':
            if linear_term is not None:
                return norm(r)**2 * coef / 2. + (linear_term * x).sum() + cons
            else:
                return norm(r)**2 * coef / 2. + cons
        elif mode == 'grad':
            if linear_term is not None:
                return coef * r + linear_term
            else:
                return coef * r
        else:
            raise ValueError("Mode incorrectly specified")

    def __repr__(self):
        return 'identity_quadratic(%f, %s, %s, %f)' % (self.coef, repr(self.center), repr(self.linear_term), self.constant_term)

    def __add__(self, other):
        """ Return an identity quadratic given by the sum in the obvious way.

        It has center of 0, would be nice to have None, but there are some
        places we are still multiplying by -1
        """
        if not (other is None or isinstance(other, identity_quadratic)):
            raise ValueError('can only add None or other identity_quadratic')


        if other is None:
            return self
        else:
            sc = self.collapsed()
            oc = other.collapsed()
            newq = identity_quadratic(sc.coef + oc.coef, 0, 
                                      sc.linear_term + oc.linear_term,
                                      sc.constant_term + oc.constant_term)
            return newq 

    def get_shapes(self):
        '''
        Determine shape of any pieces and make sure they agree
        '''
        self.zeroify()
        lt = asarray(self.linear_term)
        center = asarray(self.center)

        if lt.shape != () and center.shape != ():
            if lt.shape != center.shape:
                raise ValueError('conflicting shapes of linear_term and center')
        return lt.shape, center.shape


    def __getitem__(self, slice):
        '''
        Return a new quadratic restricted to the variables in slice
        with constant_term=0.
        '''
        lts, cts = self.get_shapes()

        if lts != ():
            lt = self.linear_term[slice]
        else:
            lt = self.linear_term

        if cts != ():
            ct = self.center[slice]
        else:
            ct = self.center
        return identity_quadratic(self.coef,
                                  ct, lt, 0)

    def collapsed(self):
        """
        Return an identity quadratic with center of 0,
        would be nice to have None, but there are some 
        places we are still multiplying by -1
        """

        if self.coef is None:
            coef = 0
        else:
            coef = self.coef

        linear_term = 0
        constant_term = self.constant_term
        if constant_term is None: 
            constant_term = 0 
        if self.center is not None:
            linear_term -= coef * self.center
            constant_term += coef * norm(self.center)**2/2.
        if self.linear_term is not None:
            linear_term += self.linear_term

        return identity_quadratic(coef, 0, linear_term, constant_term)

    def latexify(self, var=r'\beta', idx=''):
        self.zeroify()
        terms = []
        if self.coef != 0:
            if not all(self.center == 0):
                terms.append(r'\frac{L_{%s}}{2}' % idx + r'\|%s-\mu_{%s}\|^2_2' % (var, idx))
            else:
                terms.append(r'\frac{L_{%s}}{2}' % idx + r'\|%s\|^2_2' % var)
        if self.linear_term is not None and not all(self.linear_term == 0):
            terms.append(r'\left \langle \eta_{%s}, %s \right \rangle' % (idx, var))
        if self.constant_term is not None and self.constant_term != 0:
            terms.append(r'\gamma_{%s} ' % idx)
        return ' + '.join(terms)

    def _repr_latex_(self):
        return r'''\begin{equation*} %s \end{equation*} ''' % self.latexify()

    @property
    def true_center(self):
        q = self.collapsed()
        if q.coef > 0:
            return -q.linear_term / q.coef
        return None

    @property
    def conjugate(self):
        a = self.collapsed()
        if a.coef != 0:
            return identity_quadratic(1./a.coef,
                                      a.linear_term,
                                      0,
                                      -a.constant_term)
        else:
            # possible import problem here
            from ..atoms.cones import zero_constraint
            q = identity_quadratic(0,0,0,-a.constant_term)
            return zero_constraint(a.linear_term.shape, offset=-a.linear_term, quadratic=q)
