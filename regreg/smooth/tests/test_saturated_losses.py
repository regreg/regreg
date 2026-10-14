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
