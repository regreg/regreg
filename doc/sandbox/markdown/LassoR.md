```python
import numpy as np, regreg.api as rr
import rpy2.robjects as rpy2
```

# The Diabetes data from LARS

```python
rpy2.r('''
library(lars)
data(diabetes)
X = diabetes$x
Y = diabetes$y
diabetes_lars = lars(diabetes$x, diabetes$y, type='lasso')
L = diabetes_lars$lambda
''')
X = rpy.r('X')
L = rpy.r('L')
Y = rpy.r('Y')
```

```python
n, p = X.shape
n, p
```

Our loss function and penalty

```python
loss = rr.gaussian(X, Y)
loss
```

Now, our penalty:
```python
penalty = rr.l1norm(X.shape[1], lagrange=L[3])
penalty
```

Let's form the problem
```python
problem = rr.simple_problem(loss, penalty)
problem
```
and solve it

```python
beta = problem.solve(min_its=100)
beta
```

Compare this to `R`'s solution:

```python
S = rpy2.r('diabetes_lars$beta[4,]')
```

## Bound form


We can also solve this in bound form

```python
bound_form = rr.l1norm(p, bound=np.fabs(beta).sum())
bound_problem = rr.simple_problem(loss, bound_form)
bound_problem
```

Here is the solution

```python
beta_bound = bound_problem.solve(min_its=100)
beta_bound
```

