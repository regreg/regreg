from itertools import product
import nose.tools as nt

import numpy as np
import scipy.sparse

import regreg.api as rr
import regreg.affine.image2d as I2D
from regreg.identity_quadratic import identity_quadratic as sq
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_class():
    p = 50
    diff = I2D.image2d_differences((p,p+5))
    diff2 = I2D.image2d_differences((p,p+5), affine_offset=1)
    im = np.random.standard_normal((p,p+5))
    di = np.random.standard_normal(diff.output_shape)
    diff.linear_map(im)
    diff.affine_map(im)
    diff.adjoint_map(di)

    I2D.formD_smaller(p, p+5)

    np.testing.assert_allclose(diff2.affine_map(im), diff2.linear_map(im) + 1)
