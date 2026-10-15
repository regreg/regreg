"""
Regression tests for the atom bugs in gh-108 and gh-103.
"""
from copy import copy

import numpy as np
import pytest

import regreg.api as rr
import regreg.atoms.mixed_lasso as ML
from regreg.identity_quadratic import identity_quadratic as iq
from regreg.atoms.piecewise_linear import find_solution_piecewise_linear
from regreg.atoms.sparse_group_lasso import sparse_group_lasso_dual

def test_l2_epigraph_projection():
    # cone_prox had no "already in the cone" case and divided by ||v|| = 0
    K, P = rr.l2_epigraph(3), rr.l2_epigraph_polar(3)
    np.testing.assert_allclose(K.proximal(iq(1, np.array([1., 0, 5.]), 0, 0)), [1, 0, 5])
    np.testing.assert_allclose(K.cone_prox(np.array([0., 0, 1])), [0, 0, 1])
    np.testing.assert_allclose(K.cone_prox(np.array([3., 4, -6])), 0)
    np.testing.assert_allclose(P.cone_prox(np.array([1., 0, -5])), [1, 0, -5])
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.standard_normal(4) * 3
        a, b = K.cone_prox(x), P.cone_prox(x)
        np.testing.assert_allclose(a + b, x, atol=1e-12)        # Moreau
        assert abs(a @ b) < 1e-10
        assert np.linalg.norm(a[:-1]) <= a[-1] + 1e-12
        assert np.linalg.norm(b[:-1]) <= -b[-1] + 1e-12

def test_projection_complement_constraint():
    # the test was inverted: it checked the norm of the projection
    basis = np.identity(3)[:2]
    pc = rr.projection_complement((3,), basis)
    assert pc.constraint(np.array([0, 0, 1.])) == 0
    assert pc.constraint(np.array([1., 0, 0])) == np.inf
    assert pc.constraint(pc.cone_prox(np.array([1., 2, 3]))) == 0

def test_group_lasso_dual_feasibility():
    # check_feasibility was passed into the nntol slot, so groups of weight 0
    # (which must be 0 in the dual) were never flagged
    d = rr.group_lasso_dual([1, 1, 2, 2], weights={1: 0.}, lagrange=1.)
    assert d.seminorm(np.array([3., 4, .1, 0]), check_feasibility=True) == np.inf
    assert np.isfinite(d.seminorm(np.array([0., 0, .1, 0]), check_feasibility=True))
    db = rr.group_lasso_dual([1, 1, 2, 2], weights={1: 0.}, bound=1.)
    assert db.constraint(np.array([3., 4, .1, 0])) == np.inf
    assert db.constraint(np.array([0., 0, .1, 0])) == 0

def test_bound_prox_output_is_feasible():
    # constraint used `<= bound` with no tolerance, so rounding made about
    # half of the bound_prox outputs infeasible
    rng = np.random.default_rng(1)
    groups = [1, 1, 2, 2, 2, 3]
    for klass in [rr.group_lasso, rr.group_lasso_dual]:
        atom = klass(groups, bound=.7)
        for _ in range(200):
            assert atom.constraint(atom.bound_prox(rng.standard_normal(6) * 3)) == 0

def test_mixed_lasso_positive_part_prox():
    # gh-103: the POSITIVE_PART prox sent negative inputs to 0
    atom = rr.mixed_lasso(np.array([ML.POSITIVE_PART]), lagrange=1.)
    ref = rr.positive_part(1, lagrange=1.)
    for z in [-3., 0.5, 3.]:
        q = iq(1, np.array([z]), 0, 0)
        np.testing.assert_allclose(atom.proximal(q), ref.proximal(q))

def test_mixed_lasso_feasibility():
    # gh-103: NONNEGATIVE coordinates at 0 were infeasible, and POSITIVE_PART
    # coordinates (a penalty) were feasibility-checked
    m = rr.mixed_lasso(np.array([ML.NONNEGATIVE]), lagrange=1.)
    assert m.nonsmooth_objective(np.zeros(1), check_feasibility=True) == 0
    assert m.nonsmooth_objective(-np.ones(1), check_feasibility=True) == np.inf
    pp = rr.mixed_lasso(np.array([ML.POSITIVE_PART]), lagrange=1.)
    assert pp.nonsmooth_objective(-2 * np.ones(1), check_feasibility=True) == 0

def test_mixed_lasso_dual_feasibility():
    # the dual set requires u = 0 for UNPENALIZED coordinates, u >= 0 for
    # POSITIVE_PART and u <= 0 for NONNEGATIVE ones; check_feasibility was
    # also ignored by mixed_lasso_dual.seminorm
    d = rr.mixed_lasso_dual(np.array([0, 0, ML.UNPENALIZED]), 1.)
    assert d.nonsmooth_objective(np.array([.1, .1, 5.]), check_feasibility=True) == np.inf
    assert d.nonsmooth_objective(np.array([.1, .1, 0.]), check_feasibility=True) == 0
    dp = rr.mixed_lasso_dual(np.array([ML.POSITIVE_PART, ML.NONNEGATIVE]), 1.)
    assert dp.seminorm(np.array([-.5, -1.]), check_feasibility=True) == np.inf
    assert dp.seminorm(np.array([.5, 1.]), check_feasibility=True) == np.inf
    assert np.isfinite(dp.seminorm(np.array([.5, -1.]), check_feasibility=True))

def test_mixed_lasso_dual_prox_not_aliased():
    # the dual bound prox returned its work buffer, overwritten by the next call
    rng = np.random.default_rng(2)
    d = rr.group_lasso_dual([1, 1, 2, 2, 2, 3], bound=.7)
    p1 = d.proximal(iq(1, rng.standard_normal(6), 0, 0))
    saved = p1.copy()
    d.proximal(iq(1, rng.standard_normal(6), 0, 0))
    np.testing.assert_array_equal(p1, saved)

def test_piecewise_linear_zero_weights():
    # terms with weight 0 started the sweep at norm / 1e-20
    with_zero = find_solution_piecewise_linear(.5, 1, np.array([5., np.sqrt(2)]),
                                               np.array([0., np.sqrt(2)]))
    without = find_solution_piecewise_linear(.5, 1, np.array([np.sqrt(2)]),
                                             np.array([np.sqrt(2)]))
    np.testing.assert_allclose(with_zero, without)
    K = rr.group_lasso_epigraph([1, 1, 2, 2], weights={1: 0.})
    np.testing.assert_allclose(K.cone_prox(np.array([3., 4, 1, 1, .5])), [3, 4, .5, .5, 1])

def test_matrix_shaped_l1_and_sup_prox():
    # the 1-D Cython solver was given a 2-D array
    z = np.arange(6.).reshape((3, 2)) - 2
    out = rr.l1norm((3, 2), bound=.7).proximal(iq(1, z, 0, 0))
    np.testing.assert_allclose(out.ravel(), rr.l1norm(6, bound=.7).proximal(iq(1, z.ravel(), 0, 0)))
    out = rr.supnorm((3, 2), lagrange=.7).proximal(iq(1, z, 0, 0))
    np.testing.assert_allclose(out.ravel(), rr.supnorm(6, lagrange=.7).proximal(iq(1, z.ravel(), 0, 0)))

def test_sparse_group_lasso_dual_lagrange_form():
    # base_conjugate was built with lagrange=self.bound, which is None in
    # Lagrange form
    groups, w = [1, 1, 2, 2, 2], [0, .5, .2, .2, .2]
    x = np.array([1., -2, 3, .5, -1])
    d = sparse_group_lasso_dual(groups, w, lagrange=2.)
    db = sparse_group_lasso_dual(groups, w, bound=1.)
    np.testing.assert_allclose(d.nonsmooth_objective(x), 2 * db.seminorm(x, lagrange=1.))
    conj = rr.sparse_group_lasso(groups, w, bound=1.).conjugate
    np.testing.assert_allclose(conj.nonsmooth_objective(x), db.seminorm(x, lagrange=1.))

def test_block_sum_copy_and_setters():
    b = rr.block_sum(rr.l1norm, (5, 4), lagrange=1.)
    c = copy(b)
    assert type(c) is rr.block_sum and isinstance(c.atom, rr.l1norm)
    b.lagrange = 2.
    assert b.lagrange == 2.
    m = rr.block_max(rr.l1norm, (5, 4), bound=1.)
    m.bound = 3.
    assert m.bound == 3.
    # bound_prox silently returned 0 * x
    with pytest.raises(NotImplementedError):
        rr.block_sum(rr.l1norm, (5, 4), bound=1.).bound_prox(np.ones((5, 4)))
