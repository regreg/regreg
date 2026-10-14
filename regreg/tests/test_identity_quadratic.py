"""
Tests of `regreg.identity_quadratic` (gh-78).
"""
import numpy as np

from regreg.identity_quadratic import identity_quadratic as iq
from regreg.atoms.cones import zero_constraint

def test_noneify():

    q = iq(1, np.zeros(3), np.zeros(3), None)
    q.noneify()
    assert q.center is None
    assert q.linear_term is None
    assert q.constant_term == 0

    q = iq(None, np.ones(3), np.array([0, 1., 0]), 2)
    q.noneify()
    assert q.coef == 0
    np.testing.assert_allclose(q.center, np.ones(3))
    np.testing.assert_allclose(q.linear_term, [0, 1., 0])

def test_conjugate_zero_coef():
    # the zero coefficient branch used an import beyond the top-level package

    q = iq(0, 0, np.ones(3), 0)
    c = q.conjugate
    assert isinstance(c, zero_constraint)
    # the conjugate of x -> 1^T x is the indicator of {1}
    assert c.nonsmooth_objective(np.ones(3), check_feasibility=True) == 0
