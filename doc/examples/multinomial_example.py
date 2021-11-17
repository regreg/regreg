# IPython log file

import numpy as np
import regreg.smooth.mglm as M
import regreg.smooth.mglm as M
import regreg.api as rr

np.random.seed(0)
n, p = 2000, 4
Y = np.random.multinomial(1, [0.1,0.4,0.5], size=(n,))
q = Y.shape[1]

X = np.random.standard_normal((n, p))
pen = rr.l1_l2((p, q), lagrange=0.4*np.sqrt(n))
loss = M.mglm.multinomial(X, Y)
problem = rr.simple_problem(loss, pen)
problem.solve(debug=True, min_its=50, tol=1e-12)

loss_baseline = M.mglm.multinomial(X, Y, baseline=True)
pen_baseline = rr.l1_l2((p, q-1), lagrange=0.4*np.sqrt(n))
problem_baseline = rr.simple_problem(loss_baseline, pen_baseline)
problem_baseline.solve(debug=True, min_its=50, tol=1e-12)
