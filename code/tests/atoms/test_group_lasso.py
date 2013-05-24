
import numpy as np
import itertools
from copy import copy

import regreg.atoms.group_lasso as GL
import regreg.api as rr
import nose.tools as nt

from test_seminorms import Solver, all_close 

@np.testing.dec.slow
def test_proximal_maps(interactive=False):
    groups = [np.arange(10),
              np.array([1,1,2,2,2,3,3,4,4,4,4,5,5,6,6,6,6])]

    quadratic = rr.identity_quadratic(0,0,0,0)

    counter = 0
    for L, atom, q, offset, FISTA, coef_stop, G in itertools.product([0.5,1,0.1], \
                     sorted(GL.conjugate_seminorm_pairs.keys()),
                     [None, quadratic],
                     [True, False],
                     [False, True],
                     [True, False],
                     groups):

        penalty = atom(G, lagrange=L, quadratic=q)
        Z = np.random.standard_normal(penalty.shape)

        if offset:
            penalty.offset = 0.02 * np.random.standard_normal(penalty.shape)
        if q is not None:
            penalty.quadratic.linear_term = 0.02 * np.random.standard_normal(penalty.shape)

        dual = penalty.conjugate 
        yield all_close, penalty.lagrange_prox(Z, lipschitz=L), Z-dual.bound_prox(Z*L)/L, 'testing lagrange_prox and bound_prox starting from atom\n %s ' % atom, None
        # some arguments of the constructor

        nt.assert_raises(AttributeError, setattr, penalty, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, dual, 'lagrange', 4.)
        
        nt.assert_raises(AttributeError, setattr, penalty, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, dual, 'lagrange', 4.)

#         for t in Solver(penalty, Z, L, FISTA, coef_stop, interactive=interactive).all():
#             yield t

#         bound = atom(G, bound=0.3, quadratic=q)
#         if offset:
#             bound.offset = 0.02 * np.random.standard_normal(bound.shape)
#         if q is not None:
#             bound.quadratic.linear_term = 0.02 * np.random.standard_normal(bound.shape)

#         for t in Solver(bound, Z, L, FISTA, coef_stop, interactive=interactive).all():
#             yield t

    for L, atom, q, offset, FISTA, coef_stop in itertools.product( 
        [0.5,1,0.1], 
        sorted(GL.conjugate_cone_pairs.keys()),
        [None, quadratic],
        [True, False],
        [False, True],
        [False, True]):

        cone_instance = atom(G, quadratic=q)
        Z = np.random.standard_normal(cone_instance.shape)

        if offset:
            cone_instance.offset = 0.02 * np.random.standard_normal(cone_instance.shape)
        if q is not None:
            cone_instance.quadratic.linear_term = 0.02 * np.random.standard_normal(cone_instance.shape)
        for t in Solver(cone_instance, Z, L, FISTA, coef_stop).all():
            yield t


