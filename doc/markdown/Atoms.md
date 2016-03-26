# Using constraints

This tutorial illustrates how to use RegReg to solve problems using the container class. We illustrate with an example:
$$
\frac{1}{2}||y - \beta||^{2}_{2} \ \text{subject to} \  ||D\beta||_{1} \leq \delta_1,   \|\beta\|_1 \leq \delta_2
$$
This problem is solved by solving a dual problem, following the 
general derivation in the TFOCS paper
$$
\frac{1}{2}||y - D^Tu_1 - u_2||^{2}_{2} + \delta_1 \|u_1\|_{\infty} + \delta_2 \|u_2\|_{\infty}
$$
For a general loss function, the general objective has the form
$$
{\cal L}_{\epsilon}(\beta) \ \text{subject to} \  ||D\beta||_{1} \leq \delta_1,   \|\beta\|_1 \leq \delta_2
$$
which is solved by minimizing the dual
$$
{\cal L}^*_{\epsilon}(-D^Tu_1-u_2) + \delta_1 \|u_1\|_{\infty} + \delta_2 \|u_2\|_{\infty}
$$

Recall that for the sparse fused LASSO
$$
D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 
\\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)
$$
To solve this problem using RegReg we begin by loading the necessary numerical libraries

```python
%pylab inline
import numpy as np
from scipy import sparse
import regreg.api as rr
```

Next, let's generate an example signal, and solve the Lagrange
form of the problem

```python
Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = rr.signal_approximator(Y)

sparsity = rr.l1norm(len(Y), 1.4)
# TODO should make a module to compute typical Ds
D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
fused = rr.l1norm.linear(D, 25.5)
problem = rr.container(loss, sparsity, fused)

solver = rr.FISTA(problem)
solver.fit(max_its=100)
solution = problem.coefs
```

We will now solve this problem in constraint form, using the 
achieved  values $\delta_1 = \|D\widehat{\beta}\|_1, \delta_2=\|\widehat{\beta}\|_1$.
By default, the container class will try to solve this problem with the two-loop strategy.

```python
delta1 = np.fabs(D * solution).sum()
delta2 = np.fabs(solution).sum()
fused_constraint = rr.l1norm.linear(D, bound=delta1)
sparsity_constraint = rr.l1norm(500, bound=delta2)
constrained_problem = rr.container(loss, fused_constraint, sparsity_constraint)
constrained_solver = rr.FISTA(constrained_problem)
constrained_solver.composite.lipschitz = 1.01
vals = constrained_solver.fit(max_its=10, tol=1e-06, monotonicity_restart=False)
constrained_solution = constrained_solver.composite.coefs
```
We can also solve this problem approximately by smoothing one or more of the constraints with the smoothed_atom method. The smoothed constraint is then treated as a differentiable function which can be faster in some problems.

```python
smoothed_fused_constraint = fused_constraint.smoothed(rr.identity_quadratic(1e-2,0,0,0))
smoothed_constrained_problem = rr.container(loss, smoothed_fused_constraint, sparsity_constraint)
smoothed_constrained_solver = rr.FISTA(smoothed_constrained_problem)
vals = smoothed_constrained_solver.fit(tol=1e-06)
smoothed_constrained_solution = smoothed_constrained_solver.composite.coefs
print np.linalg.norm(solution - constrained_solution) / np.linalg.norm(solution)
print np.linalg.norm(solution - smoothed_constrained_solution) / np.linalg.norm(solution)
```

```python
import numpy as np
from scipy import sparse

import regreg.api as rr

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = rr.signal_approximator(Y)

sparsity = rr.l1norm(len(Y), 1.4)
# TODO should make a module to compute typical Ds
D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
fused = rr.l1norm.linear(D, 25.5)
problem = rr.container(loss, sparsity, fused)

solver = rr.FISTA(problem)
solver.fit(max_its=100)
solution = solver.composite.coefs

delta1 = np.fabs(D * solution).sum()
delta2 = np.fabs(solution).sum()

fused_constraint = rr.l1norm.linear(D, bound=delta1)
sparsity_constraint = rr.l1norm(500, bound=delta2)

constrained_problem = rr.container(loss, fused_constraint, sparsity_constraint)
constrained_solver = rr.FISTA(constrained_problem)
constrained_solver.composite.lipschitz = 1.01
vals = constrained_solver.fit(max_its=10, tol=1e-06, monotonicity_restart=False)
constrained_solution = constrained_solver.composite.coefs

fused_constraint = rr.l1norm.linear(D, bound=delta1)
smoothed_fused_constraint = fused_constraint.smoothed(rr.identity_quadratic(1e-2,0,0,0))
smoothed_constrained_problem = rr.container(loss, smoothed_fused_constraint, sparsity_constraint)
smoothed_constrained_solver = rr.FISTA(smoothed_constrained_problem)
vals = smoothed_constrained_solver.fit(tol=1e-06)
smoothed_constrained_solution = smoothed_constrained_solver.composite.coefs

#pylab.clf()
pylab.scatter(np.arange(Y.shape[0]), Y,c='red', label=r'$Y$')
pylab.plot(solution, c='yellow', linewidth=5, label='Lagrange')
pylab.plot(constrained_solution, c='green', linewidth=3, label='Constrained')
pylab.plot(smoothed_constrained_solution, c='black', linewidth=1, label='Smoothed')
pylab.legend()
#pylab.plot(conjugate_coefs, c='black', linewidth=3)	
#pylab.plot(conjugate_coefs_gen, c='gray', linewidth=1)		
```


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

# Mixing penalties and constraints


This tutorial illustrates how to use RegReg to solve problems that have seminorms in both the objective and the contraint. We illustrate with an example:

$$
\frac{1}{2}||y - \beta||^{2}_{2} + \lambda \|\beta\|_1 \text{ subject to} \  ||D\beta||_{1} \leq \delta   
$$
with
$$
D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 
0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)
$$

To solve this problem using RegReg we begin by loading the necessary numerical libraries

```python
%pylab inline
import numpy as np
from scipy import sparse
import regreg.api as rr
```
Next, let's generate an example signal, and solve the Lagrange form of the problem

```python
Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = rr.signal_approximator(Y)

sparsity = rr.l1norm(len(Y), lagrange=1.4)
D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
fused = rr.l1norm.linear(D, lagrange=25.5)
problem = rr.container(loss, sparsity, fused)

solver = rr.FISTA(problem)
solver.fit(max_its=100, tol=1e-10)
solution = solver.composite.coefs
```

We will now solve this problem in constraint form, using the 
achieved  value :math:`\delta = \|D\widehat{\beta}\|_1`.
By default, the container class will try to solve this problem with the two-loop strategy.

```python
delta = np.fabs(D * solution).sum()
sparsity = rr.l1norm(len(Y), lagrange=1.4)
fused_constraint = rr.l1norm.linear(D, bound=delta)
constrained_problem = rr.container(loss, fused_constraint, sparsity)
constrained_solver = rr.FISTA(constrained_problem)
constrained_solver.composite.lipschitz = 1.01
constrained_solver.perform_backtrack = False
vals = constrained_solver.fit(max_its=10, tol=1e-06, monotonicity_restart=False)
constrained_solution = constrained_solver.composite.coefs
```

We can now check that the obtained value matches the constraint,

```python
constrained_delta = np.fabs(D * constrained_solution).sum()
print delta, constrained_delta
print np.linalg.norm(solution - constrained_solution) / np.linalg.norm(solution)
```

```python
import numpy as np
from scipy import sparse
import regreg.api as R

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = R.signal_approximator(Y)

sparsity = R.l1norm(len(Y), lagrange=1.4)
D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
fused = R.l1norm.linear(D, lagrange=25.5)
problem = R.container(loss, sparsity, fused)

solver = R.FISTA(problem)
solver.fit(max_its=100, tol=1e-10)
solution = solver.composite.coefs

delta = np.fabs(D * solution).sum()
sparsity = R.l1norm(len(Y), lagrange=1.4)
fused_constraint = R.l1norm.linear(D, bound=delta)
constrained_problem = R.container(loss, fused_constraint, sparsity)
constrained_solver = R.FISTA(constrained_problem)
constrained_solver.composite.lipschitz = 1.01
constrained_solver.perform_backtrack = False
vals = constrained_solver.fit(max_its=10, tol=1e-06, monotonicity_restart=False)
constrained_solution = constrained_solver.composite.coefs

constrained_delta = np.fabs(D * constrained_solution).sum()
print delta, constrained_delta

pylab.scatter(np.arange(Y.shape[0]), Y)
pylab.plot(solution, c='y', linewidth=3)	
pylab.plot(constrained_solution, c='r', linewidth=1)

```
