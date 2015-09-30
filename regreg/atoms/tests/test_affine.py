from __future__ import print_function, division, absolute_import

from copy import copy
import nose.tools as nt

import numpy as np

import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_class():

    n, p = (10, 5)
    D = np.random.standard_normal((n,p))
    v = np.random.standard_normal(n)
    pen = rr.l1norm.affine(D, v, lagrange=0.4)

    pen2 = rr.l1norm(n, lagrange=0.4, offset=np.random.standard_normal(n))
    pen2.quadratic = None
    cls = type(pen)
    pen_aff = cls(pen2, rr.affine_transform(D, v))

    for _pen in [pen, pen_aff]:
        # Run to ensure code gets executed in tests (smoke test)
        print(_pen.dual)
        print(_pen.latexify())
        print(str(_pen))
        print(repr(_pen))
        print(_pen._repr_latex_())
        _pen.nonsmooth_objective(np.random.standard_normal(p))
        q = rr.identity_quadratic(0.5,0,0,0)
        smoothed_pen = _pen.smoothed(q)
