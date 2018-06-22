"""
Simple ADMM implementation
"""

import numpy as np
from regreg.smooth import sum as smooth_sum

class admm_problem(object):

    """
    Objective function is the loss plus the atom composed with the transform.

    .. math::

        \ell(x) + {\cal P}(Dx + \alpha)

    Our transform is a map $\beta \mapsto D\beta + \alpha, and we introduce new variables
    $z=D\beta+\alpha$ so our affine constraint is $Dx-z=-\alpha$.

    In ADMM slides p.14 of https://stanford.edu/class/ee364b/lectures/admm_slides.pdf
    we have $A,B,c=D,-I,-\alpha$.

    """

    def __init__(self, 
                 loss, 
                 atom, 
                 transform, 
                 augmented_param, 
                 quadratic=None):
        (self.loss,
         self.atom,
         self.transform,
         self.augmented_param,
         self.quadratic) = (loss,
                            atom,
                            transform,
                            augmented_param,
                            quadratic)

        self.loss_coef = self.loss.coef                        # x in ADMM notes
        self.atom_coef = np.zeros(self.transform.output_shape) # z in ADMM notes
        self.dual_coef = np.zeros(self.transform.output_shape) # y in ADMM notes
        self.augmented_param = augmented_param                 # rho in ADMM notes

        linear = 
        Q = composition
        qloss = quadratic
        self.augmented_loss = smooth_sum(self.loss,
                                         quadratic_loss)

        assert (self.loss.shape == self.transform.output_shape)

    def update_loss_coef(self):
        """
        Minimize over x: self.loss(x) + rho/2 x^TD^TDx + x^T D^T(y -rho * (z - \alpha))
        """
        
    def update_atom_coef(self):
        """
        Minimize over z: atom(z) + rho/2 \|z-Dx-\alpha\|^2_2 - z^Ty
        """
        rho = self.augmented_param
        center_term = self.transform.affine_map(self.loss_coef)
        linear_term = -self.dual_coef
        Q = identity_quadratic(rho, center_term, linear_term, 0)
        self.atom_coef[:] = self.atom.proximal_map(Q)

    def update_dual_coef(self):
        """
        Dual update
        """
        rho = self.augmented_param
        self.dual_coef[:] += rho * (self.transform.affine_map(self.loss_coef) - self.atom_coef)

def test_admm(n=100, p=10):

    import regreg.api as rr, regreg.affine as ra
    import numpy as np

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    loss = rr.squared_error(X, Y)
    D = np.identity(p)
    pen = rr.l1norm(p, lagrange=1.5)

    ADMM = admm_problem(loss, pen, ra.astransform(D), 0.5)

if __name__ == "__main__":

    test_admm()
