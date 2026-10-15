"""
Regression tests for atom methods that could not run (gh-61, gh-62).
"""
import numpy as np
import pytest

import regreg.api as rr
from regreg.atoms.simplex import simplex_constraint, maximum
from regreg.atoms.linear_constraints import projection, projection_complement
from regreg.atoms.group_lasso import check_KKT
from regreg.problems.composite import smooth_conjugate

def _atoms():
    basis = np.identity(4)[:2]
    return [rr.l1norm(4, lagrange=1.),
            rr.supnorm(4, bound=1.),
            rr.l2norm(4, lagrange=1.),
            rr.positive_part(4, lagrange=1.),
            rr.nonnegative(4),
            rr.zero_constraint(4),
            rr.l1_epigraph(5),
            rr.group_lasso([0, 0, 1, 1], lagrange=1.),
            rr.group_lasso_dual([0, 0, 1, 1], lagrange=1.),
            rr.l1_l2((3, 4), lagrange=1.),
            rr.linf_linf((3, 4), lagrange=1.),
            rr.l1_l1((3, 4), lagrange=1.),
            rr.nuclear_norm((3, 4), lagrange=1.),
            rr.operator_norm_epigraph((3, 4)),
            projection((4,), basis),
            projection_complement((4,), basis),
            rr.sparse_group_lasso([0, 0, 1, 1], np.ones(4), lagrange=1.),
            rr.sparse_group_block((3, 4), 1, 2, lagrange=1.),
            ]

@pytest.mark.parametrize('atom', _atoms(), ids=lambda a: type(a).__name__)
def test_get_dual(atom):
    # get_dual overrides called the `dual` property and raised TypeError
    transform, conj = atom.get_dual()
    transform2, conj2 = atom.dual
    assert type(conj) == type(conj2)
    assert conj == atom.conjugate

def test_properties_use_overrides():
    # `conjugate` and `dual` were bound to base class functions, so
    # subclass overrides of get_conjugate / get_dual never ran
    class my_l1norm(rr.l1norm):
        def get_conjugate(self):
            return 'conjugate'
        def get_dual(self):
            return 'dual'
    pen = my_l1norm(4, lagrange=1.)
    assert pen.conjugate == 'conjugate'
    assert pen.dual == 'dual'

def test_projection_conjugate():
    # with the properties dispatching to the overrides, projection uses
    # linear_constraint.get_conjugate, which keeps the basis
    basis = np.identity(4)[:2]
    for klass, other in [(projection, projection_complement),
                         (projection_complement, projection)]:
        conj = klass((4,), basis).conjugate
        assert type(conj) == other
        np.testing.assert_array_equal(conj.basis, basis)

def test_block_prox():
    # block_sum.lagrange_prox and block_max.bound_prox used xrange
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4))
    v = rr.block_sum(rr.l1norm, (5, 4), lagrange=0.5).lagrange_prox(X, lipschitz=2.)
    np.testing.assert_allclose(v, np.sign(X) * np.maximum(np.fabs(X) - 0.25, 0))
    v = rr.block_max(rr.l1norm, (5, 4), bound=1.).bound_prox(X)
    np.testing.assert_allclose(v, [rr.l1norm(4, bound=1.).bound_prox(x) for x in X])

@pytest.mark.parametrize('make_atom', [
    lambda q: rr.slope([3, 2, 1], lagrange=1., quadratic=q),
    lambda q: rr.nuclear_norm_epigraph((3, 4), quadratic=q),
    lambda q: simplex_constraint(4, quadratic=q),
    lambda q: maximum(4, quadratic=q),
    lambda q: projection((4,), np.identity(4)[:2], quadratic=q),
    ])
def test_smooth_conjugate_imported(make_atom):
    # the conjugate of an atom with a quadratic is a smooth_conjugate, which
    # several modules did not import
    q = rr.identity_quadratic(1., 0, 0, 0)
    atom = make_atom(q)
    conj = atom.conjugate
    assert isinstance(conj, smooth_conjugate)
    # Fenchel equality: the gradient of the conjugate at v attains the sup
    # in sup_x v^T x - atom(x)
    v = np.linspace(-1, 1, int(np.prod(atom.shape))).reshape(atom.shape)
    value, grad = conj.smooth_objective(v, 'both')
    np.testing.assert_allclose(value,
                               (v * grad).sum() - atom.nonsmooth_objective(grad),
                               atol=1e-6)

@pytest.mark.parametrize('klass', [simplex_constraint, maximum])
def test_simplex_linear(klass):
    # linear() did not import linear_transform, passed an unknown keyword and
    # built the atom on the input shape
    rng = np.random.default_rng(0)
    D = rng.standard_normal((3, 5))
    offset = rng.standard_normal(3)
    x = rng.standard_normal(5)
    composed = klass.linear(D, offset=offset)
    atom = klass(3, offset=-offset)
    assert composed.atom.shape == (3,)
    assert np.isclose(composed.nonsmooth_objective(x),
                      atom.nonsmooth_objective(D.dot(x)))

def test_projection_linear():
    basis = np.identity(3)[:2]
    D = np.random.default_rng(0).standard_normal((3, 5))
    composed = projection.linear(D, basis)
    assert composed.atom.shape == (3,)
    np.testing.assert_array_equal(composed.atom.basis, basis)

def test_group_lasso_check_KKT():
    # check_KKT had a typo'd variable name and np.nozero
    pen = rr.group_lasso([0, 0, 1, 1], weights={0: 1., 1: 1.}, lagrange=1.)
    solution = np.array([1., 1, 0, 0])
    grad = -np.array([1., 1, 0.5, 0]) / np.sqrt(2)
    assert not check_KKT(pen, grad, solution, 1.)
    grad[2:] = -2
    assert check_KKT(pen, grad, solution, 1.)
