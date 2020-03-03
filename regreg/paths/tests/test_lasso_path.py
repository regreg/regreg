import numpy as np, regreg.api as rr, regreg.affine as ra
import nose.tools as nt
import regreg.paths.lasso as lasso

from ...tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_path(n=200,p=50):
    '''
    run a basic path algorithm

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    lasso1 = lasso.lasso_path.gaussian(X, Y, np.ones(X.shape[1]), nstep=23)
    sol1 = lasso1.main(inner_tol=1.e-5)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_unpenalized(n=200, p=50):
    '''
    run a basic path algorithm with some unpenalized variables

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    weights = np.ones(X.shape[1])
    weights[0:2] = 0
    lasso1 = lasso.lasso_path.gaussian(X, 
                                       Y, 
                                       weights,
                                       nstep=23)
    sol1 = lasso1.main(inner_tol=1.e-5)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_elastic_net(n=200, p=50):
    '''
    run a basic elastic net path algorithm

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    weights = np.ones(X.shape[1])
    lasso1 = lasso.lasso_path.gaussian(X, 
                                       Y, 
                                       weights,
                                       nstep=23,
                                       alpha=0.5,
                                       elastic_net_param=np.ones(X.shape[1]))
    sol1 = lasso1.main(inner_tol=1.e-9)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_elastic_net_unpenalized(n=200, p=50):
    '''
    run a basic elastic net path algorithm with unpenalized
    

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    enet = np.ones(X.shape[1])
    enet[0] = 0
    weights = np.ones(X.shape[1])
    weights[0] = 0

    lasso1 = lasso.lasso_path.gaussian(X, 
                                       Y, 
                                       weights,
                                       nstep=23,
                                       alpha=0.5,
                                       elastic_net_param=enet)
    sol1 = lasso1.main(inner_tol=1.e-9)
    beta1 = sol1['beta']

