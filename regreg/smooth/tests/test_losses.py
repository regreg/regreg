import numpy as np
import nose.tools as nt

from regreg.smooth.losses import huberized_svm

def test_huberized_svm():

    Y = np.random.binomial(1,0.5,size=(10,))
    X = np.random.standard_normal((10,5))

    L = huberized_svm(X, Y, 0.01)
    L.smooth_objective(np.zeros(L.shape), 'both')

    L.latexify()
