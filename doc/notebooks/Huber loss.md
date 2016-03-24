# Huberized lasso tutorial

The Huberized lasso minimizes the following objective
$$
H_{\delta}(Y - X\beta) + \lambda \|\beta\|_1
$$
where $H_{\delta}(\cdot)$ is a function applied element-wise,
$$
 H_{\delta}(r) = \left\{\begin{array}{ll} r^2/2 & \mbox{ if } |r| \leq \delta \\ \delta r - \delta^2/2 & \mbox{ else}\end{array} \right.
$$
To solve this problem using RegReg we begin by loading the necessary numerical libraries

```python
import numpy as np
import regreg.api as rr
```

Next, let's generate some example data,

```python
X = np.random.normal(0,1,500000).reshape((500,1000))
Y = np.random.randint(0,2,500)
```
Now we can create the problem object, beginning with the loss function

```python
penalty = rr.l1norm(1000,lagrange=5.)
loss = rr.l1norm.affine(X,-Y, lagrange=1.).smoothed(rr.identity_quadratic(1,0,0,0))
```
The penalty contains the regularization parameter that can be easily accessed and changed,

```python
penalty.lagrange
```
Now we can create the final problem object

```python
problem = rr.container(loss, penalty)
```
Next, we can select our algorithm of choice and use it solve the problem,

```python
solver = rr.FISTA(problem)
obj_vals = solver.fit(max_its=200, tol=1e-6)
solution = solver.composite.coefs
(solution != 0).sum()
```
Here max_its represents primal iterations, and tol is the primal tolerance. 

```python
obj_vals
```
