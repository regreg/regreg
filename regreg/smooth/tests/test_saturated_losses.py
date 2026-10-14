"""
Regression tests for case weights, offsets and values of saturated losses.
"""
import numpy as np

from .. import glm, mglm
from ..binary import (logistic_loglike,
                      probit_loglike,
                      cloglog_loglike,
                      huber_svm,
                      _huber_svm)
from ..losses import huberized_svm

def _binary_losses(n, **kw):
    y = (np.arange(n) % 2).astype(float)
    return [logistic_loglike((n,), y, **kw),
            probit_loglike((n,), y, **kw),
            cloglog_loglike((n,), y, **kw),
            glm.poisson_loglike((n,), np.arange(n), **kw)]

def test_case_weights_not_modified():
    # gh-69: losses used to multiply the caller's case weights in place

    n = 5
    X = np.random.standard_normal((n, 2))
    Y = np.random.standard_normal(n)
    w = np.arange(1., n + 1)
    G = glm.glm(X, Y, glm.gaussian_loglike((n,), Y, case_weights=w))
    values = [G.objective(np.ones(2)) for _ in range(3)]
    np.testing.assert_allclose(values, values[0])
    np.testing.assert_allclose(G.case_weights, np.ones(n))
    np.testing.assert_allclose(G.saturated_loss.case_weights, w)

    eta = np.linspace(-1, 1, n)
    for L in _binary_losses(n, case_weights=w) + [glm.gaussian_loglike((n,), Y, case_weights=w)]:
        incoming = np.linspace(1, 2, n)
        incoming_cp = incoming.copy()
        f1 = L.smooth_objective(eta, 'func', case_weights=incoming)
        h1 = L.hessian(eta, case_weights=incoming)
        f2 = L.smooth_objective(eta, 'func', case_weights=incoming)
        h2 = L.hessian(eta, case_weights=incoming)
        np.testing.assert_allclose(incoming, incoming_cp)
        np.testing.assert_allclose(L.case_weights, w)
        np.testing.assert_allclose(f1, f2)
        np.testing.assert_allclose(h1, h2)

def test_cox_integer_case_weights():
    # gh-69: in-place multiply failed for integer weights

    X = np.random.standard_normal((5, 2))
    G = glm.glm.cox(X, np.random.exponential(size=5), np.ones(5),
                    case_weights=np.ones(5, int))
    G0 = glm.glm.cox(X, G.saturated_loss.event_times, np.ones(5))
    np.testing.assert_allclose(G.objective(np.zeros(2)), G0.objective(np.zeros(2)))

def test_stacked_common_case_weights():
    # gh-69: weights compounded across the component losses

    n, k = 6, 3
    Y = np.random.standard_normal((n, k))
    w = np.arange(1., n + 1)
    v = np.linspace(1, 2, n)
    losses = [glm.gaussian_loglike((n,), Y[:,i], case_weights=v) for i in range(k)]
    L = mglm.stacked_common_loglike(losses, case_weights=w)
    eta = np.random.standard_normal((n, k))
    expected = sum([0.5 * np.sum(w * v * (eta[:,i] - Y[:,i])**2) for i in range(k)])
    for _ in range(2):
        np.testing.assert_allclose(L.smooth_objective(eta, 'func'), expected)
    np.testing.assert_allclose(L.case_weights, w)
    for l in losses:
        np.testing.assert_allclose(l.case_weights, v)

def test_offsets():
    # gh-70: offset ignored in some objectives and in all Hessians

    n = 5
    eta = np.linspace(-1, 1, n)
    off = np.full(n, 0.7)
    for L_off, L in zip(_binary_losses(n, offset=off), _binary_losses(n)):
        np.testing.assert_allclose(L_off.smooth_objective(eta, 'func'),
                                   L.smooth_objective(eta - off, 'func'))
        np.testing.assert_allclose(L_off.smooth_objective(eta, 'grad'),
                                   L.smooth_objective(eta - off, 'grad'))
        np.testing.assert_allclose(L_off.hessian(eta),
                                   L.hessian(eta - off))

    X = np.random.standard_normal((n, 2))
    y = (np.arange(n) % 2).astype(float)
    beta, o = np.array([.3, -.2]), np.array([.5, 1.])
    np.testing.assert_allclose(glm.glm.logistic(X, y, offset=o).hessian(beta),
                               glm.glm.logistic(X, y).hessian(beta - o))

def test_huberized_svm_coef_offset():
    # gh-70: huberized_svm ignored both coef and offset

    X = np.random.standard_normal((6, 2))
    labels = np.array([0, 1, 1, 0, 1, 0])
    beta, o = np.array([.3, -.2]), np.array([10., 10.])
    L = huberized_svm(X, labels, .5)
    f, g = L.smooth_objective(beta, 'both')

    f5, g5 = huberized_svm(X, labels, .5, coef=5.).smooth_objective(beta, 'both')
    np.testing.assert_allclose(f5, 5 * f)
    np.testing.assert_allclose(g5, 5 * g)

    np.testing.assert_allclose(huberized_svm(X, labels, .5, offset=o).smooth_objective(beta, 'func'),
                               L.smooth_objective(beta - o, 'func'))

    # agrees with the saturated huber_svm loss composed with X
    np.testing.assert_allclose(huber_svm((6,), labels, .5).smooth_objective(X.dot(beta), 'func'), f)

def test_poisson_saturated_constant():
    # gh-71: loss at the saturated fit plus the constant should be 0

    y = np.array([0., 1, 2, 5])
    eta = np.log(np.where(y > 0, y, 1e-12))
    for coef in [1., 3.]:
        P = glm.poisson_loglike((4,), y, coef=coef)
        np.testing.assert_allclose(P.smooth_objective(eta, 'func') + P.quadratic.constant_term,
                                   0, atol=1e-10)

def test_huber_svm_gradient():
    # gh-71: value of _huber_svm did not match its gradient

    z = np.array([-1., .2, .8, 1.5])
    eps, d = .5, 1e-6
    numerical = (_huber_svm(z + d, eps)[0] - _huber_svm(z - d, eps)[0]) / (2 * d)
    np.testing.assert_allclose(_huber_svm(z, eps)[1], numerical, atol=1e-6)

def test_logistic_hessian_overflow():
    # gh-71: overflow branch clamped the wrong entries

    L = logistic_loglike((3,), np.array([0., 1, 1]))
    H = L.hessian(np.array([200., 0., 1.]))
    p = 1 / (1 + np.exp(-1.))
    np.testing.assert_allclose(H[1:], [0.25, p * (1 - p)])
    assert 0 <= H[0] < 1e-80

def test_huber_copy_subsample_case_weights():
    # gh-73: __copy__ dropped case_weights, so subsample replaced them

    from copy import copy
    w = np.array([1., 0, 2, 3])
    for L in [glm.huber_loss((4,), np.arange(4.), .5, case_weights=w),
              huber_svm((4,), np.array([0, 1, 1, 0]), .5, case_weights=w)]:
        np.testing.assert_allclose(copy(L).case_weights, w)
        S = L.subsample([0, 2, 3])
        np.testing.assert_allclose(S.case_weights, [1., 0, 2, 3])
        S = L.subsample([0, 1])
        np.testing.assert_allclose(S.case_weights, [1., 0, 0, 0])
        np.testing.assert_allclose(L.case_weights, w)

def test_multinomial_case_weights():
    # gh-73: multinomial losses ignored their stored case weights

    C = np.eye(3)[[0, 1, 2, 0, 1]]
    w = np.array([1., 0, 2, .5, 3])
    eta3 = np.random.standard_normal((5, 3))
    eta2 = eta3[:, :2]
    for klass, eta in [(mglm.multinomial_loglike, eta3),
                       (mglm.multinomial_baseline_loglike, eta2)]:
        zero = klass(eta.shape, C, case_weights=np.zeros(5))
        np.testing.assert_allclose(zero.smooth_objective(eta, 'func'), 0, atol=1e-12)

        stored = klass(eta.shape, C, case_weights=w)
        passed = klass(eta.shape, C)
        f1, g1 = stored.smooth_objective(eta, 'both')
        f2, g2 = passed.smooth_objective(eta, 'both', case_weights=w)
        np.testing.assert_allclose(f1, f2)
        np.testing.assert_allclose(g1, g2)

        S = passed.subsample([0, 2])
        keep = np.array([1., 0, 1, 0, 0])
        np.testing.assert_allclose(S.smooth_objective(eta, 'func'),
                                   passed.smooth_objective(eta, 'func', case_weights=keep))
