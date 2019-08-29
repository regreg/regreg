"""
Simple ADMM implementation
"""

import numpy as np
from ..smooth import sum as smooth_sum
from ..smooth.quadratic import quadratic_loss
from ..affine import aslinear, astransform
from ..identity_quadratic import identity_quadratic

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
                 quadratic=None,
                 fit_args={}):

        (self.loss,
         self.atom,
         self.transform,
         self.augmented_param,
         self.quadratic) = (loss,
                            atom,
                            astransform(transform),
                            augmented_param,
                            quadratic)

        self.loss_coefs = self.loss.coefs                       # x in ADMM notes
        self.dual_coefs = np.zeros(self.transform.output_shape) # y in ADMM notes
        self.atom_coefs = np.zeros(self.transform.output_shape) # z in ADMM notes
        self.augmented_param = augmented_param                  # rho in ADMM notes

        self.linear_transform = aslinear(self.transform)                # D
        qloss = quadratic_loss.squared_transform(self.linear_transform) # x^TD^TDx / 2
        qloss.coef *= self.augmented_param                              # scale by rho
        self.augmented_loss = smooth_sum([self.loss,
                                          qloss])

        self.fit_args = fit_args # for the smooth_sum FISTA run

        assert (self.loss.shape == self.transform.output_shape)

    def update_loss_coefs(self):
        """
        Minimize over x: self.loss(x) + rho/2 x^TD^TDx + x^T D^T(y -rho * (z - \alpha))
        """
        y, z, rho, alpha = (self.dual_coefs,
                            self.atom_coefs,
                            self.augmented_param,
                            self.transform.affine_offset)
        if alpha is None:
            alpha = 0
        linear_term = self.linear_transform.dot(y - rho * (z - alpha))
        self.loss_coefs[:] = self.augmented_loss.solve(quadratic=identity_quadratic(0, 0, linear_term, 0),
                                                       **self.fit_args)

    def update_atom_coefs(self):
        """
        Minimize over z: atom(z) + rho/2 \|z-Dx-\alpha\|^2_2 - z^Ty
        """
        rho = self.augmented_param
        center_term = self.transform.affine_map(self.loss_coefs)
        linear_term = -self.dual_coefs
        Q = identity_quadratic(rho, center_term, linear_term, 0)
        self.atom_coefs[:] = self.atom.proximal(Q)

    def update_dual_coefs(self):
        """
        Dual update
        """
        rho = self.augmented_param
        self.dual_coefs[:] += rho * (self.transform.affine_map(self.loss_coefs) - self.atom_coefs)

    def solve(self, niter=20):
        for _ in range(niter):
            self.update_loss_coefs()
            self.update_atom_coefs()
            self.update_dual_coefs()

