import numpy as np
import regreg.api as rr
import nose.tools as nt
from ...tests.decorators import set_seed_for_test

@set_seed_for_test()
@np.testing.dec.skipif(True, msg='NESTA path not implemented correctly')
def test_nesta_path():

    def atom_factory(candidate_set):
        return rr.nonnegative.linear(constraint_matrix[:,candidate_set])

    Z2 = np.zeros((100,8))
    Z2[:,:3] = X[:,:3]
    Z2[:,3:6] = -X[:,:3]
    Z2[:,6:] = -X[:,3:]

    constraint_matrix = np.zeros((3,9))
    constraint_matrix[2,1:6] = 1
    constraint_matrix[0,6] = 1
    constraint_matrix[1,7] = 1
    constraint = rr.nonnegative.linear(constraint_matrix)

    lasso_constraint = rr.nesta_path.squared_error(Z2, Y, atom_factory, nstep=10)
    sol4 = lasso_constraint.main()
