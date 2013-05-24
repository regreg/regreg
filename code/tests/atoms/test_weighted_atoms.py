import numpy as np
import regreg.api as rr
import itertools
from numpy import testing as npt

import regreg.atoms.weighted_atoms as WA

from test_seminorms import Solver

def test_proximal_maps():
    shape = 20

    bound = 0.14
    lagrange = 0.13

    Z = np.random.standard_normal(shape) * 2
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    quadratic = rr.identity_quadratic(0,0,W,0)

    basis = np.linalg.svd(np.random.standard_normal((4,20)), full_matrices=0)[2]

    w1 = np.ones(20) * 0.5
    w2 = w1 * 0
    w2[:10] = 2.

    for L, atom, q, offset, FISTA, coef_stop, weights in itertools.product([0.5,1,0.1], \
                                              sorted(WA.conjugate_weighted_pairs.keys()),
                                              [None, quadratic],
                                              [True,False],
                                              [False, True],
                                              [False, True],
                                              [w1, w2]):

        # we only have two weighted atoms,
        # l1 in lagrange and supnorm in bound

        print 'weights: ', weights.shape
        if atom == WA.l1norm:
            penalty = atom(shape, weights, quadratic=q,
                           offset=offset, lagrange=lagrange)
        else:
            penalty = atom(shape, weights, quadratic=q,
                           offset=offset, bound=bound)
            
        Z = np.random.standard_normal(penalty.shape)

        if offset:
            penalty.offset = 0.02 * np.random.standard_normal(penalty.shape)
        if q is not None:
            penalty.quadratic.linear_term = 0.02 * np.random.standard_normal(penalty.shape)
        for t in Solver(penalty, Z, L, FISTA, coef_stop).all():
            yield t


def test_weighted_l1():
    a =rr.weighted_l1norm(10, 2*np.ones(10), lagrange=0.5)
    b= rr.l1norm(10, lagrange=1)
    z = np.random.standard_normal(10)
    npt.assert_equal(b.lagrange_prox(z), a.lagrange_prox(z))
    npt.assert_equal(b.dual[1].bound_prox(z), a.dual[1].bound_prox(z))

def test_weighted_l1_with_zero():
    z = np.random.standard_normal(5)
    a=rr.weighted_l1norm(5, weights=[0,1,1,1,1], lagrange=0.5)
    b=a.dual[1]
    c=rr.l1norm(4, lagrange=0.5)
    npt.assert_equal(a.lagrange_prox(z), z-b.bound_prox(z))
    npt.assert_equal(a.lagrange_prox(z)[0], z[0])
    npt.assert_equal(a.lagrange_prox(z)[1:], c.lagrange_prox(z[1:]))
