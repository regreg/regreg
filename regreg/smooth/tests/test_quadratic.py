"""
Tests of `regreg.smooth.quadratic` (gh-74).
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from ..quadratic import quadratic_loss, squared_error, signal_approximator
from ...identity_quadratic import identity_quadratic as iq

def _total(L, x):
    # the loss including its quadratic term
    return L.smooth_objective(x, 'func') + L.quadratic.objective(x, 'func')

def _cases():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    dense = A @ A.T + np.eye(3)
    quad = iq(.7, np.array([.3, .1, -.4]), np.array([1., 0, -1]), 1.3)
    yield dict(shape=(3,), Q=np.array([1., 2, 3]), Qdiag=True)
    yield dict(shape=(3,), Q=np.array([1., 2, 3]), Qdiag=True, coef=2.5,
               offset=np.array([1., -2, .5]), quadratic=quad)
    yield dict(shape=(3,), Q=dense)
    yield dict(shape=(3,), Q=dense, coef=1.7, offset=np.array([.4, -1, 2]),
               quadratic=quad)

@pytest.mark.parametrize('kw', list(_cases()))
def test_conjugate(kw):
    # compare with sup_x u^T x - f(x), computed numerically

    L = quadratic_loss(**kw)
    C = L.get_conjugate(factor=True)
    for u in [np.array([.3, -1., 2.]), np.array([-2., .5, 0.])]:
        opt = minimize(lambda x: _total(L, x) - u @ x, np.zeros(3),
                       method='BFGS', options={'gtol': 1e-10})
        np.testing.assert_allclose(_total(C, u), -opt.fun, atol=1e-8)
        # the gradient of the conjugate is the maximizer
        np.testing.assert_allclose(C.smooth_objective(u, 'grad'), opt.x, atol=1e-5)

def test_conjugate_errors():

    dense = np.diag([1., 2, 3])
    with pytest.raises(ValueError):
        quadratic_loss((3,), Q=dense).get_conjugate(factor=False)
    transform = quadratic_loss.squared_transform(np.identity(3))
    with pytest.raises(ValueError):
        transform.get_conjugate(factor=True)

def test_objective_vars_not_shared():
    # squared_error / signal_approximator used to modify the class attribute

    before = dict(quadratic_loss.objective_vars)
    S = squared_error(np.ones((3, 2)), np.ones(3))
    A = signal_approximator(np.ones(3))
    assert quadratic_loss.objective_vars == before
    assert S.atom.objective_vars['offset'] == 'Y'
    assert A.objective_vars['offset'] == 'Y'
    assert quadratic_loss((3,)).objective_vars['offset'] == before['offset']
