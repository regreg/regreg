# Multinomial regression

The multinomial regression problem minimizes the objective

$$
-2\left[ \sum_{j=1}^{J-1} \sum_{k=1}^p \beta_{jk}\sum_{i=1}^n x_{ik}y_{ij}
 - \sum_{i=1}^n \log \left(1 + \mbox{exp}(x_i^T\beta_j) \right)\right]
$$
which corresponds to a baseline category logit model for $J$ nominal categories (e.g. Agresti, p.g. 272). For $i \ne J$ the probabilities are measured relative to a baseline category $J$
$$
\frac{P(\mbox{Category } i)}{P(\mbox{Category } J)} = \mbox{logit}(x^T\beta_i) = \frac{1}{1 + \mbox{exp}(-x^T\beta_i)}
$$

To solve this problem using RegReg we begin by loading the necessary numerical libraries

```python
import numpy as np
import regreg.api as rr
```

The only code needed to add multinomial regression to RegReg is a class
with one method which computes the objective and its gradient.


Next, let's generate some example data. The multinomial counts will be stored in a $n \times J$ array

```python
J = 5
n = 1000
p = 50
X = np.random.standard_normal((n,p))
Y = np.random.randint(0,10,n*J).reshape((n,J))
```

Now we can create the problem object, beginning with the loss function. The coefficients will be stored in a $p \times J-1$ array, and we need to let RegReg know that the coefficients will be a 2d array instead of a vector. We can do this by defining the input_shape in a linear_transform object that multiplies by X,

```python
multX = rr.linear_transform(X, input_shape=(p,J-1))
loss = rr.multinomial_deviance.linear(multX, counts=Y)
```

Next, we can solve the problem
```python
problem = rr.container(loss)
solver = rr.FISTA(problem)
solver.fit()
```

When $J=2$ this model should reduce to logistic regression. We can easily check that this is the case by first fitting the multinomial model


```python
J = 2
Y = np.random.randint(0,10,n*J).reshape((n,J))
multX = rr.linear_transform(X, input_shape=(p,J-1))	
loss = rr.multinomial_deviance.linear(multX, counts=Y)
problem = rr.container(loss)
solver = rr.FISTA(problem)
solver.fit(tol=1e-6)
multinomial_coefs = solver.composite.coefs.flatten()
```

and then the equivalent logistic regresison model

```python
successes = Y[:,0]
trials = np.sum(Y, axis=1)
loss = rr.logistic_deviance.linear(X, successes=successes, trials=trials)
problem = rr.container(loss)
solver = rr.FISTA(problem)
solver.fit(tol=1e-6)
logistic_coefs = solver.composite.coefs
```

Finally we can check that the two models gave the same coefficients

```python
print np.linalg.norm(multinomial_coefs - logistic_coefs) / np.linalg.norm(logistic_coefs)
```

