import numpy as np
from scipy import sparse
from algorithms import FISTA, ISTA

class seminorm(object):
    """
    A seminorm container class for storing/combining seminorm_atom classes
    """
    def __init__(self, *atoms):
        self.atoms = []
        self.primal_dim = -1
        self.segments = []
        idx = 0
        for atom in atoms:
            if self.primal_dim < 0:
                self.primal_dim = atom.p
            else:
                if atom.p != self.primal_dim:
                    raise ValueError("primal dimensions don't agree")
            self.atoms.append(atom)
            self.segments.append(slice(idx, idx+atom.m))
            idx += atom.m
        self.total_dual = idx

    def __add__(self,y):
        #Combine two seminorms
        def atoms():
            for obj in [self, y]:
                for atom in obj.atoms:
                    yield atom
        return seminorm(*atoms())

    def evaluate(self, x):
        out = 0.
        for atom in self.atoms:
            out += atom.evaluate(x)
        return out
    
    def evaluate_dual(self, u):
        out = 0.
        for atom, segment in zip(self.atoms, self.segments):
            out += atom.evaluate_dual(u[segment])
        return out
    
    def dual_prox(self, u, L_D=1.):
        """
        Return (unique) minimizer

        .. math::

           v^{\lambda}(u) = \text{argmin}_{v \in \real^m} \frac{1}{2}
           \|v-u\|^2_2  s.t.  h^*_i(v) \leq \infty, 0 \leq i \leq M-1

        where *m*=u.shape[0]=np.sum(self.dual_dims), :math:`M`=self.M
        and :math:`h^*_i` is the conjugate of 
        self.atoms[i].l * self.atoms[i].evaluate and 
        :math:`\lambda_i`=self.atoms[i].l.

        This is used in the inner loop with :math:`u=z-g/L` when finding
        self.primal_prox, i.e., the signal approximator problem.
        """
        v = np.empty(u.shape)
        for atom, segment in zip(self.atoms, self.segments):
            v[segment] = atom.dual_prox(u[segment], L_D)
        return v

    default_solver = FISTA
    def primal_prox(self, y, L_P=1, with_history=False, debug=False, max_its=5000, tol=1e-14):
        """
        The proximal function for the primal problem
        """
        yL = L_P * y
        if not hasattr(self, 'dualopt'):
            self.dualp = self.dual_problem(yL, L_P=L_P)
            #Approximate Lipschitz constant
            self.dualp.L = 1.1*self.power_LD(debug=debug)
            self.dualopt = seminorm.default_solver(self.dualp)
            self.dualopt.debug = debug
        self._dual_prox_center = yL
        history = self.dualopt.fit(max_its=max_its, min_its=5, tol=tol, backtrack=False)
        if with_history:
            return self.primal_from_dual(y, self.dualopt.problem.coefs/L_P), history
        else:
            return self.primal_from_dual(y, self.dualopt.problem.coefs/L_P)

    def power_LD(self,max_its=50,tol=1e-5, debug=False):
        """
        Approximate the Lipschitz constant for the dual problem using power iterations
        """
        v = np.random.standard_normal(self.primal_dim)
        z = np.zeros(self.total_dual)
        old_norm = 0.
        norm = 1.
        itercount = 0
        while np.fabs(norm-old_norm)/norm > tol and itercount < max_its:
            z *= 0.
            for atom, segment in zip(self.atoms, self.segments):
                z[segment] += atom.multiply_by_D(v)
            v *= 0.
            for atom, segment in zip(self.atoms, self.segments):
                v += atom.multiply_by_DT(z[segment])
            old_norm = norm
            norm = np.linalg.norm(v)
            v /= norm
            if debug:
                print "L", norm
            itercount += 1
        return norm
        #return np.sqrt(norm)

    def primal_from_dual(self, y, u):
        """
        Calculate the primal coefficients from the dual coefficients
        """
        x = y * 1.
        for atom, segment in zip(self.atoms, self.segments):
            x -= atom.multiply_by_DT(u[segment])
        return x

    def dual_problem(self, y, L_P=1, initial=None):
        """
        Return a problem instance of the dual
        prox problem with a given y value.
        """
        self._dual_prox_center = y
        if initial is None:
            z = np.random.standard_normal(self.total_dual)
            initial = self.dual_prox(z, 1.)
        nonsmooth = self.evaluate_dual
        prox = self.dual_prox
        return dummy_problem(self._dual_smooth_eval, nonsmooth, prox, initial, 1.)

    def _dual_smooth_eval(self,v,mode='both'):

        """
        The smooth component and/or gradient of the dual objective        
        """
        
        primal = self.primal_from_dual(self._dual_prox_center, v)

        if mode == 'func':
            return (primal**2).sum() / 2.
        elif mode == 'both' or mode == 'grad':
            g = np.zeros(self.total_dual)
            for atom, segment in zip(self.atoms, self.segments):
                g[segment] = -atom.multiply_by_D(primal)
            if mode == 'grad':
                return g
            if mode == 'both':
                return (primal**2).sum() / 2., g
        else:
            raise ValueError("Mode not specified correctly")

    def problem(self, smooth_eval, smooth_multiplier=1., initial=None):
        prox = self.primal_prox
        nonsmooth = self.evaluate
        if initial is None:
            initial = np.random.standard_normal(self.primal_dim)
        if self.evaluate(initial) + smooth_eval(initial,mode='func') == np.inf:
            raise ValueError('initial point is not feasible')
        
        return dummy_problem(smooth_eval, nonsmooth, prox, initial, smooth_multiplier)



class dummy_problem(object):
    """
    A generic way to specify a problem
    """
    def __init__(self, smooth_eval, nonsmooth, prox, initial, smooth_multiplier=1):
        # Do we need to store this?
        #self.initial = initial.copy()
        self.coefs = initial.copy()
        self.obj_rough = nonsmooth
        self._smooth_eval = smooth_eval
        self._prox = prox
        self.smooth_multiplier = smooth_multiplier


    def smooth_eval(self, x, mode='both'):
        output = self._smooth_eval(x, mode=mode)
        if mode == 'both':
            return self.smooth_multiplier * output[0], self.smooth_multiplier * output[1]
        elif mode == 'grad' or mode == 'func':
            return self.smooth_multiplier * output
        else:
            raise ValueError("Mode incorrectly specified")

    def obj(self, x):
        return self.smooth_eval(x,mode='func') + self.obj_rough(x)

    def proximal(self, x, g, L, prox_control=None):
        """
        Compute the proximal optimization

        prox_control: If not None, then a dictionary of parameters for the prox procedure
        """
        z = x - g / L
        if prox_control is None:
            return self._prox(z, L)
        else:
            return self._prox(z, L, **prox_control)
