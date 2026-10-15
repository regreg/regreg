import numpy as np

from copy import copy
import scipy.optimize

import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_quadratic():

    l = rr.quadratic_loss(5, coef=3., offset=np.arange(5))
    l.quadratic = rr.identity_quadratic(1, np.ones(5), 2*np.ones(5), 3.)
    c1 = l.get_conjugate()

    q1 = rr.identity_quadratic(3, np.arange(5), 0, 0)
    q2 = q1 + l.quadratic
    c2 = rr.zero(5, quadratic=q2.collapsed()).conjugate

    ww = np.random.standard_normal(5)
    np.testing.assert_almost_equal(c2.smooth_objective(ww, 'grad'),
                                   c1.smooth_objective(ww, 'grad'))

    np.testing.assert_almost_equal(c2.objective(ww),
                                   c1.objective(ww))

    np.testing.assert_almost_equal(c2.smooth_objective(ww, 'func') + 
                                   c2.nonsmooth_objective(ww),
                                   c1.smooth_objective(ww, 'func') + 
                                   c1.nonsmooth_objective(ww))


def _brute_conjugate(sc, v):
    # sup_u v^T u - h(u) - q(u), with h = sc and q = sc.quadratic
    f = lambda u: (sc.smooth_objective(u, 'func') + sc.quadratic.objective(u, 'func') - v @ u)
    return -scipy.optimize.minimize(f, np.zeros(v.shape), method='BFGS',
                                    options={'gtol': 1e-10}).fun

def test_conjugate_with_quadratic():
    # gh-83: the conjugate of a smooth_conjugate with a quadratic added
    # failed in both branches

    rng = np.random.default_rng(0)
    p = 4
    iq = rr.identity_quadratic
    atom = rr.l1norm(p, lagrange=.7, offset=rng.standard_normal(p),
                     quadratic=iq(.3, 0, rng.standard_normal(p), .4))
    smoothing = iq(.5, rng.standard_normal(p), rng.standard_normal(p), -.2)
    for q in [iq(0, 0, rng.standard_normal(p), .6),                       # r = 0
              iq(2., rng.standard_normal(p), rng.standard_normal(p), -.3)]: # r > 0
        sc = rr.smooth_conjugate(copy(atom), smoothing)
        sc.quadratic = q
        G = sc.conjugate
        for v in [.5 * rng.standard_normal(p), rng.standard_normal(p)]:
            if q.coef == 0:
                value = G.nonsmooth_objective(v)
            else:
                value = G.smooth_objective(v, 'func')
                # gradient by finite differences
                e = 1e-6
                num = np.array([(G.smooth_objective(v + e * u, 'func') -
                                 G.smooth_objective(v - e * u, 'func')) / (2 * e)
                                for u in np.identity(p)])
                np.testing.assert_allclose(G.smooth_objective(v, 'grad'), num, atol=1e-6)
            np.testing.assert_allclose(value, _brute_conjugate(sc, v), atol=1e-8)
