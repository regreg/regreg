# Smoothing the seminorm tutorial

This tutorial illustrates the FUSED LASSO signal approximator problem
and the use of smoothing the seminorm as in NESTA and TFOCS (TODO put links to these papers)

The sparse fused lasso minimizes the objective

$$
\frac{1}{2}||y - \beta||^{2}_{2} + \lambda_{1}||D\beta||_{1} + \lambda_2 \|\beta\|_1
$$

$$
D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 
0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)
$$

To solve this problem using RegReg we begin by loading the necessary numerical libraries. 

```python
import numpy as np
%pylab inline
from scipy import sparse

import regreg.api as R

# generate the data

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
```

Now we can create the problem object, beginning with the loss function

```python
loss = R.signal_approximator(Y)
sparsity = R.l1norm(len(Y), lagrange=1.8)

# fused
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D = sparse.csr_matrix(D)
fused = R.l1norm.linear(D, lagrange=25.5)
```

The penalty can be smoothed to create a 
smooth function object which can be solved with FISTA.

```python
Q = R.identity_quadratic(0.01, 0, 0, 0)
smoothed_sparsity = sparsity.smoothed(Q)
smoothed_fused = fused.smoothed(Q)
```

The smoothing is defined by
$$
\begin{aligned}
h^{\epsilon}_{K}(D\beta+\alpha) &= \sup_{u \in K} u^T(D\beta+\alpha) - \frac{\epsilon}{2}\|u\|^2_2 \\
&= \epsilon \left(\|(D\beta+\alpha)/\epsilon\|^2_2 - \|(D\beta+\alpha)/\epsilon-P_K((D\beta+\alpha)/\epsilon)\|^2_2\right)
\end{aligned}
$$
with gradient
$$
\nabla_{\beta} h^{\epsilon}_{K}(D\beta+\alpha) = D^TP_K((D\beta+\alpha)/\epsilon)
$$

Finally, we can create the final problem object,

```python
problem = R.smooth_sum([loss, smoothed_sparsity, smoothed_fused])
solver = R.FISTA(problem)
%timeit solver.fit()
```

which has both the loss function and the seminorm represented in it. 
We will estimate $\beta$ for various values of $\epsilon$:

```python
solns = []
for eps in [.5**i for i in range(15)]:
    Q = R.identity_quadratic(eps, 0, 0, 0)
    smoothed_sparsity = sparsity.smoothed(Q)
    smoothed_fused = fused.smoothed(Q)
    problem = R.smooth_sum([loss, smoothed_sparsity, smoothed_fused])
    solver = R.FISTA(problem)
    solver.fit()
    solns.append(solver.composite.coefs.copy())
    pylab.plot(solns[-1])
```

We can then plot solution to see the result of the regression,

```python
import numpy as np
import pylab	
from scipy import sparse

import regreg.api as R

# generate the data

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

loss = R.signal_approximator(Y)
sparsity = R.l1norm(len(Y), lagrange=1.8)

# fused
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D
D = sparse.csr_matrix(D)
fused = R.l1norm.linear(D, lagrange=25.5)

Q = R.identity_quadratic(0.01, 0, 0, 0)
smoothed_sparsity = sparsity.smoothed(Q)
smoothed_fused = fused.smoothed(Q)

problem = R.smooth_sum([loss, smoothed_sparsity, smoothed_fused])
solver = R.FISTA(problem)

solns = []
pylab.scatter(range(500), Y)
for eps in [.5**i for i in range(15)]:
   Q = R.identity_quadratic(eps, 0, 0, 0)
   smoothed_sparsity = sparsity.smoothed(Q)
   smoothed_fused = fused.smoothed(Q)
   problem = R.smooth_sum([loss, smoothed_sparsity, smoothed_fused])
   solver = R.FISTA(problem)
   solver.fit()
   solns.append(solver.composite.coefs.copy())
   pylab.plot(solns[-1])
```
