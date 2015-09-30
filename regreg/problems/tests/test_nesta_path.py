import numpy as np
import regreg.api as rr
import nose.tools as nt
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_lasso_path():

    X = np.random.standard_normal((100,5))
    Z = np.zeros((100,10))
    Y = np.random.standard_normal(100)
    Z[:,5:] = -X
    Z[:,:5] = X

    lasso1 = rr.lasso.squared_error(X,Y, nstep=12)
    lasso2 = rr.lasso.squared_error(Z,Y, positive_part=np.arange(10), nstep=12)

    sol1 = lasso1.main()
    beta1 = sol1['beta'].todense()[1:]

    sol2 = lasso2.main()
    beta2 = sol2['beta'].todense()[1:]
    beta2 = beta2[:5] - beta2[5:]

    nt.assert_true(np.linalg.norm(beta1-beta2) < 1.e-3 * np.linalg.norm(beta1))
    Z2 = np.zeros((100,8))
    Z2[:,:3] = X[:,:3]
    Z2[:,3:6] = -X[:,:3]
    Z2[:,6:] = -X[:,3:]
    lasso3 = rr.lasso.squared_error(Z2,Y, positive_part=np.arange(6), nstep=12)
    sol3 = lasso3.main()

    beta3 = sol3['beta'].todense()[1:]

    newbeta3 = np.zeros_like(beta2)
    newbeta3[:3] = beta3[:3] - beta3[3:6]
    newbeta3[3:] = -beta3[6:]
    nt.assert_true(np.linalg.norm(beta1-newbeta3) < 1.e-3 * np.linalg.norm(beta1))

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
