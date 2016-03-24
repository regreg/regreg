# Poisson regression tutorial

The Poisson regression problem minimizes the objective
$$
-2 \left(Y^TX\beta - \sum_{i=1}^n \mbox{exp}(x_i^T\beta) \right), \qquad Y_i \in {0,1,2,\ldots}
$$
which corresponds to the usual Poisson regression model
$$
P(Y_i=j) = \frac{\mbox{exp}(jx_i^T\beta-\mbox{exp}(x_i^T\beta))}{j!}
$$

To solve this problem using RegReg we begin by loading the necessary numerical libraries

```python
import numpy as np
import regreg.api as rr
```
The only code needed to add Poisson regression is a class
with one method which computes the objective and its gradient.

Next, let's generate some example data,

```python
n = 1000
p = 50
X = np.random.standard_normal((n,p))
Y = np.random.randint(0,100,n)
```
Now we can create the problem object, beginning with the loss function

```python
loss = rr.poisson_deviance.linear(X, counts=Y)
```
Next, we can fit this model in the usual way

```python
problem = rr.container(loss)
solver = rr.FISTA(problem)
obj_vals = solver.fit()
solution = solver.composite.coefs
```

