import numpy as np
from regreg.affine.multiscale import multiscale

def _multiscale_matrix(p, minsize=None):
    minsize = minsize or int(p**(1/3.))
    rows = []
    for i in range(p):
        for j in range(i, p):
            if (j - i) >= minsize:
                row = np.zeros(p)
                row[i:j] = 1. / (j-i)
                rows.append(row)
    return np.array(rows)

def test_multiscale():

    M = _multiscale_matrix(200)
    Mtrans = multiscale(200)
    V = np.random.standard_normal(M.shape[1])
    W = np.random.standard_normal(M.shape[0])

    np.testing.assert_allclose(np.dot(M, V), Mtrans.linear_map(V))
    np.testing.assert_allclose(np.dot(M.T, W), Mtrans.adjoint_map(W))


