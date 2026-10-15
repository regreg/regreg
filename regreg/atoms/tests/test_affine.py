from __future__ import print_function, division, absolute_import

from copy import copy

import numpy as np

import regreg.api as rr

def test_class():

    rng = np.random.default_rng(1200)
    n, p = (10, 5)
    D = rng.standard_normal((n,p))
    v = rng.standard_normal(n)
    pen = rr.l1norm.affine(D, v, lagrange=0.4)

    pen2 = rr.l1norm(n, lagrange=0.4, offset=rng.standard_normal(n))
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
        _pen.nonsmooth_objective(rng.standard_normal(p))
        q = rr.identity_quadratic(0.5,0,0,0)
        smoothed_pen = _pen.smoothed(q)
