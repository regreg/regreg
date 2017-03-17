

```python
import regreg.api as rr
import numpy as np
import rpy2.robjects as rpy2
```


```python
rpy2.r('library(lars); data(diabetes)')
X = np.asarray(rpy2.r('diabetes$x'))
Y = np.asarray(rpy2.r('diabetes$y'))
```


```python
X = np.hstack([X, np.ones((X.shape[0],1))])
```


```python
l2 = rr.l2norm.affine(X,-Y,bound=0.65*np.linalg.norm(Y))
l1 = rr.l1norm(X.shape[1], lagrange=1)
l2s = l2.smoothed(rr.identity_quadratic(1.e-12,0,0,0))
problem = rr.simple_problem(l2s, l1)
```


```python
primal_nesta, dual_nesta = rr.nesta(None, l1, l2)
np.linalg.norm(Y - np.dot(X, primal_nesta)) / np.linalg.norm(Y)
```


```python
l1_lagrange = rr.l1norm(X.shape[1],lagrange=np.fabs(primal_nesta).sum())
loss = rr.squared_error(X,Y, coef=2)
newsoln = rr.simple_problem(loss, l1_lagrange).solve()
np.linalg.norm(Y - np.dot(X,newsoln)) / np.linalg.norm(Y)
```


```python
transform, atom = l2.dual
primal_tfocs, dual_tfocs = rr.tfocs(l1, transform, atom)
np.linalg.norm(Y - np.dot(X, primal_tfocs)) / np.linalg.norm(Y)
```


```python
# %timeit primal_tfocs, dual_tfocs = rr.tfocs(l1, transform, atom)
```


```python
# %timeit primal_nesta, dual_nesta = rr.nesta(None, l1, l2)
```


```python
np.linalg.norm(primal_tfocs - primal_nesta) / (1+np.linalg.norm(primal_nesta))

```


```python
np.linalg.norm(dual_tfocs - dual_nesta) / (1+np.linalg.norm(dual_nesta))
```

## Noiseless case: minimimum L1 norm reconstruction


```python
n, p = 200, 5000
X = np.random.standard_normal((n, p))
l1 = rr.l1norm(p, lagrange=1)
beta = np.zeros(p)
beta[:10] = 10
Y = np.dot(X, beta)

constraint = rr.zero_constraint.affine(X,-Y)
transform, atom = constraint.dual
primal_tfocs, dual_tfocs = rr.tfocs(l1, transform, atom)



```


```python
np.linalg.norm(Y - np.dot(X, primal_tfocs)) / np.linalg.norm(Y)

```


```python

```


```python
primal_tfocs[:20]
```


```python

```


```python

```
