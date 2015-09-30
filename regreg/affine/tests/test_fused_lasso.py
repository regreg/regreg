from itertools import product
import nose.tools as nt

import numpy as np
import scipy.sparse

import regreg.api as rr
import regreg.affine.fused_lasso as FL
from regreg.identity_quadratic import identity_quadratic as sq
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_class():
    p = 50
    for order in range(1,3):
        fused = FL.trend_filter.grid(p, order=order)
        fused2 = FL.trend_filter(np.arange(p), order=order)
        V = np.random.standard_normal(p)
        U = np.random.standard_normal(p - order)
        np.testing.assert_allclose(fused.linear_map(V), fused2.linear_map(V))
        np.testing.assert_allclose(fused.affine_map(V), fused2.affine_map(V))
        np.testing.assert_allclose(fused.adjoint_map(U), fused2.adjoint_map(U))

        V2 = np.random.standard_normal((p, 3))
        U2 = np.random.standard_normal((p - order, 3))

        np.testing.assert_allclose(fused.linear_map(V2), fused2.linear_map(V2))
        np.testing.assert_allclose(fused.affine_map(V2), fused2.affine_map(V2))
        np.testing.assert_allclose(fused.adjoint_map(U2), fused2.adjoint_map(U2))

        if order == 1:
            fusedI = FL.trend_filter_inverse.grid(p, order=order)
            fusedI2 = FL.trend_filter_inverse(np.arange(p), order=order)

            np.testing.assert_allclose(fusedI.linear_map(U), fusedI2.linear_map(U))
            np.testing.assert_allclose(fusedI.affine_map(U), fusedI2.affine_map(U))
            np.testing.assert_allclose(fusedI.adjoint_map(V), fusedI2.adjoint_map(V))

            np.testing.assert_allclose(fusedI.linear_map(U2), fusedI2.linear_map(U2))
            np.testing.assert_allclose(fusedI.affine_map(U2), fusedI2.affine_map(U2))
            np.testing.assert_allclose(fusedI.adjoint_map(V2), fusedI2.adjoint_map(V2))


def test_difference_transform():
    p = 50
    for order in range(1,3):
        FL.difference_transform(np.arange(p), order=order, sorted=False)
        FL.difference_transform(np.arange(p), order=order, transform=False)
