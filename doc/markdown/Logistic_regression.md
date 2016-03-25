# Regularized logistic regression tutorial

The $\ell_2$ regularized logistic regression problem minimizes the objective

$$
-2\left(Y^TX\beta - \sum_i \log \left[ 1 + \exp(x_i^T\beta) \right] \right) + \lambda \|\beta\|_2^2
$$
which corresponds to the usual logistic regression model

$$
P(Y_i=1) = \mbox{logit}(x_i^T\beta) = \frac{1}{1 + \mbox{exp}(-x_i^T\beta)}
$$

To solve this problem using RegReg we begin by loading the necessary numerical libraries

```python
import numpy as np
import regreg.api as rr
```
The only code needed to add logistic regression is a class
with one method which computes the objective and its gradient.

.. literalinclude:: ../code/regreg/smooth.py
..   :pyobject: logistic_deviance
   

Next, let's generate some example data,

```python
X = np.random.normal(0,1,500000).reshape((500,1000))
Y = np.random.randint(0,2,500)
```
Now we can create the problem object, beginning with the loss function

```python
loss = rr.logistic_deviance.linear(X,successes=Y)
penalty = rr.identity_quadratic(1., 0., 0., 0.)
loss.quadratic = penalty
loss
```
The logistic log-likelihood function is written without a matrix :math:`X`. We use the ".linear" to specify the linear composition :math:`X\beta`. Similarly, we could use ".affine" to specify an offset :math:`X\beta + \alpha`.
The penalty contains the regularization parameter that can be easily accessed and changed,

```python
penalty.coef
```

Next, we can select our algorithm of choice and use it solve the problem,

```python
solver = rr.FISTA(loss)
obj_vals = solver.fit(max_its=100, tol=1e-5)
solution = solver.composite.coefs
```
Here max_its represents primal iterations, and tol is the primal tolerance. 

```python
obj_vals
```

