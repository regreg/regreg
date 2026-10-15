import numpy as np

from regreg.smooth.losses import huberized_svm

def test_huberized_svm():

    rng = np.random.default_rng(1900)
    Y = rng.binomial(1,0.5,size=(10,))
    X = rng.standard_normal((10,5))

    L = huberized_svm(X, Y, 0.01)
    L.smooth_objective(np.zeros(L.shape), 'both')

    L.latexify()
