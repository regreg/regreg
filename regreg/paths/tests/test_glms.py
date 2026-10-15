import numpy as np

from .. import lasso, warm_start

def test_gaussian(n=200,p=50):
    '''
    run a basic path algorithm

    '''
    rng = np.random.default_rng(3100)
    X = rng.standard_normal((n,p))
    Y = rng.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + rng.standard_normal(n)

    lasso1 = lasso.gaussian(X, Y, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        lagrange_proportion=0.2,
                                                        nstep=5) # initialized at "null" model
    sol1 = warm_start(lasso1, 
                      lagrange_sequence, 
                      (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                      inner_tol=1.e-5)
    beta1 = sol1['beta']

def test_huber(n=200,p=50):
    '''
    run a basic path algorithm

    '''
    rng = np.random.default_rng(3101)
    X = rng.standard_normal((n,p))
    Y = rng.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + rng.standard_normal(n)

    lasso1 = lasso.huber(X, Y, 0.1, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        lagrange_proportion=0.2,
                                                        nstep=5) # initialized at "null" model
    sol1 = warm_start(lasso1, 
                      lagrange_sequence, 
                      (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                      inner_tol=1.e-5)
    beta1 = sol1['beta']

def test_logistic(n=200,p=50):
    '''
    run a basic path algorithm

    '''
    rng = np.random.default_rng(3102)
    X = rng.standard_normal((n,p))
    Y = rng.binomial(1, 0.5, size=(n,))

    lasso1 = lasso.logistic(X, Y, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        lagrange_proportion=0.2,
                                                        nstep=5) # initialized at "null" model
    sol1 = warm_start(lasso1, 
                      lagrange_sequence, 
                      (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                      inner_tol=1.e-5)
    beta1 = sol1['beta']

def test_huber_svm(n=200,p=50):
    '''
    run a basic path algorithm

    '''
    rng = np.random.default_rng(3103)
    X = rng.standard_normal((n,p))
    Y = rng.binomial(1, 0.5, size=(n,))

    lasso1 = lasso.huber_svm(X, Y, 0.1, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        lagrange_proportion=0.2,
                                                        nstep=5) # initialized at "null" model
    sol1 = warm_start(lasso1, 
                      lagrange_sequence, 
                      (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                      inner_tol=1.e-5)
    beta1 = sol1['beta']

def test_poisson(n=200,p=50):
    '''
    run a basic path algorithm

    '''
    rng = np.random.default_rng(3104)
    X = rng.standard_normal((n,p))
    Y = rng.poisson(5, size=(n,))

    lasso1 = lasso.poisson(X, Y, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        lagrange_proportion=0.2,
                                                        nstep=5) # initialized at "null" model
    sol1 = warm_start(lasso1, 
                      lagrange_sequence, 
                      (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                      inner_tol=1.e-5)
    beta1 = sol1['beta']

def test_cox(n=200,p=50):
    '''
    run a basic path algorithm

    '''
    rng = np.random.default_rng(3105)
    X = rng.standard_normal((n,p))
    T = rng.exponential(5, size=(n,))
    C = rng.binomial(1, 0.5, size=(n,))
    lasso1 = lasso.cox(X, T, C, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        lagrange_proportion=0.2,
                                                        nstep=5) # initialized at "null" model
    sol1 = warm_start(lasso1, 
                      lagrange_sequence, 
                      (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                      inner_tol=1.e-5)
    beta1 = sol1['beta']
