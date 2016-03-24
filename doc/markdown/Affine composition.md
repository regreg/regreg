# Adding affine offsets to seminorms

This tutorial illustrates how to add
an affine part to the seminorm.
Suppose that instead of shrinking the values in the fused LASSO to 0,
we want to shrink them all towards a given vector $\alpha$

This can be achieved formally  sparse fused lasso minimizes the objective

$$
\frac{1}{2}||y - \beta||^{2}_{2} + \lambda_{1}||D\beta||_{1} + \lambda_2 \|\beta-\alpha\|_1
$$
with

$$
D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 
0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)
$$

Everything is roughly the same as in the fused LASSO, we just need
to change the second seminorm to have this affine offset.

```python
%pylab inline
import numpy as np
from scipy import sparse

import regreg.api as R
# set the seed, for reproducibility
np.random.seed(40)
```
Let's generate the same example signal,

```python
Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
```

Now we can create the problem object, beginning with the loss function

```python
alpha = np.linspace(0,10,500)
Y += alpha
loss = R.signal_approximator(Y)

shrink_to_alpha = R.l1norm(Y.shape, offset=-alpha, lagrange=3.)
shrink_to_alpha
```
which creates an affine_atom object with $\lambda_2=3$. That is, it creates the penalty
$$
3 \|\beta-\alpha\|_{1}
$$

that will be added to a smooth loss function.
Next, we create the fused lasso matrix and the associated l1norm object,

```python
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D
D = sparse.csr_matrix(D)
fused = R.l1norm.linear(D, lagrange=25.5)
```
Here we first created D, converted it a sparse matrix, and then created an l1norm object with the sparse version of D and $\lambda_1 = 25.5$. 
Finally, we can create the final problem object, and solve it.

```python
cont = R.container(loss, shrink_to_alpha, fused)
solver = R.FISTA(cont)
# This problem seems to get stuck restarting
%timeit solver.fit(max_its=200, tol=1e-10)
solution = cont.coefs
```
We can then plot solution to see the result of the regression,

```python
import numpy as np
import pylab	
from scipy import sparse

np.random.seed(40)
import regreg.api as R

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

alpha = np.linspace(0,10,500)
Y += alpha
loss = R.signal_approximator(Y)

shrink_to_alpha = R.l1norm(Y.shape, offset=-alpha, lagrange=3.)

D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D = sparse.csr_matrix(D)
fused = R.l1norm.linear(D, lagrange=25.5)

cont = R.container(loss, shrink_to_alpha, fused)
solver = R.FISTA(cont)
solver.debug = True
solver.fit(max_its=200, tol=1e-10)
solution = solver.composite.coefs


pylab.clf()
pylab.plot(solution, c='g', linewidth=6, label=r'$\hat{Y}$')	
pylab.plot(alpha, c='black', linewidth=3, label=r'$\alpha$')	
pylab.scatter(np.arange(Y.shape[0]), Y, facecolor='red', label=r'$Y$')
pylab.legend()


pylab.gca().set_xlim([0,650])
pylab.legend()
```
