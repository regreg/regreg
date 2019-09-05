.. _normalize:

Using an affine transform to normalize a matrix :math:`X`
---------------------------------------------------------

This tutorial illustrates how to use an affine transform to normalize a
data matrix :math:`X` without actually storing the normalized matrix.

Suppose that we would like to solve the LASSO

.. math::

   \frac{1}{2}||y - X\beta||^{2}_{2} + \lambda||\beta||_{1}

after :math:`X` has been normalized to have column mean 0 and standard
deviation 1.

The Diabetes data from LARS
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To begin, let's grab the diabetes data from the lars package in R.

.. nbplot::
   :format: python

    >>> import rpy2.robjects as rpy2
    >>> import numpy as np
    >>> import regreg.api as rr
    >>> rpy2.r('''
    >>> suppressMessages(library(lars))
    >>> data(diabetes)
    >>> X = diabetes$x
    >>> Y = diabetes$y
    >>> ''')
    >>> X = np.asarray(rpy2.r('X'))
    >>> Y = np.asarray(rpy2.r('Y'))
    >>> n, p = X.shape

We can always manually center and scale the columns of :math:`X`

.. nbplot::
   :format: python

    >>> Xnorm = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
    >>> print(np.mean(Xnorm, axis=0))
    >>> print(np.std(Xnorm, axis=0))

    [ -8.03781375e-18  -4.01890688e-17  -8.03781375e-18   2.00945344e-17
      -1.60756275e-17  -2.81323481e-17   0.00000000e+00  -5.62646963e-17
       2.00945344e-17   0.00000000e+00]
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]

However if :math:`X` is very large we may not want to store the
normalized copy. This is especially true if :math:`X` is sparse because
centering the columns will likely make the matrix dense. Instead we can
use the normalize affine transformation

.. nbplot::
   :format: python

    >>> Xnorm_rr = rr.normalize(X, center=True, scale=True) # the default

We can verify that multiplications with Xnorm\_rr are done correctly

.. nbplot::
   :format: python

    >>> test_vec1 = np.random.standard_normal(p)
    >>> test_vec2 = np.random.standard_normal(n)
    >>> print(np.linalg.norm(np.dot(Xnorm, test_vec1) - Xnorm_rr.linear_map(test_vec1)))
    >>> print(np.linalg.norm(np.dot(Xnorm, test_vec1) - Xnorm_rr.dot(test_vec1)))
    >>> print(np.linalg.norm(np.dot(Xnorm.T, test_vec2) - Xnorm_rr.adjoint_map(test_vec2)))
    >>> print(np.linalg.norm(np.dot(Xnorm.T, test_vec2) - Xnorm_rr.T.dot(test_vec2)))

    1.79514995551e-14
    1.79514995551e-14
    4.35469325497e-14
    4.35469325497e-14

Finally, we can solve the LASSO with both matrices and see that the
solutions are the same,

.. nbplot::
   :format: python

    >>> loss1 = rr.squared_error(Xnorm, Y)
    >>> sparsity = rr.l1norm(p, lagrange = 800.)
    >>> problem1 = rr.container(loss1, sparsity)
    >>> solver1 = rr.FISTA(problem1)
    >>> solver1.fit()
    >>> coefs1 = solver1.composite.coefs
    >>>
    >>> loss2 = rr.squared_error(Xnorm_rr, Y)
    >>> problem2 = rr.container(loss2, sparsity)
    >>> solver2 = rr.FISTA(problem2)
    >>> solver2.fit()
    >>> coefs2 = solver2.composite.coefs

.. nbplot::
   :format: python

    >>> print(np.linalg.norm(coefs1-coefs2))

    5.03747716687e-14

.. nbplot::
   :format: python

    >>> coefs2
    array([ -0.        ,  -7.5600115 ,  25.04059284,  13.31564146,
            -0.72193921,  -1.92904486, -10.70843844,   0.        ,
            22.67989762,   1.37154407])

.. nbplot::
   :format: python

    >>> coefs1
    array([ -0.        ,  -7.5600115 ,  25.04059284,  13.31564146,
            -0.72193921,  -1.92904486, -10.70843844,   0.        ,
            22.67989762,   1.37154407])


