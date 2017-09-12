.. _diabetes_example:

A numerical comparison to R

The Diabetes data from LARS
---------------------------

.. nbplot::

    >>> import numpy as np, regreg.api as rr
    >>> import rpy2.robjects as rpy2

Let's grab the diabetes data from the lars package in R

.. nbplot::

    >>> rpy2.r('''
    >>> library(lars)
    >>> data(diabetes)
    >>> X = diabetes$x
    >>> Y = diabetes$y
    >>> diabetes_lars = lars(diabetes$x, diabetes$y, type='lasso')
    >>> L = diabetes_lars$lambda
    >>> ''')
    >>> X = rpy2.r('X')
    >>> L = rpy2.r('L')
    >>> Y = rpy2.r('Y')
    

.. nbplot::

    >>> X = np.asarray(X)
    >>> Y = np.asarray(Y)
    >>>
    >>> n, p = X.shape
    >>> n, p
    (442, 10)

Our loss function and penalty

.. nbplot::

    >>> loss = rr.glm.gaussian(X, Y)
    >>> loss



.. math::

    \ell^{\text{Gauss}}\left(X_{}\beta\right)


Now, our penalty:

.. nbplot::

    >>> penalty = rr.l1norm(X.shape[1], lagrange=L[3])
    >>> penalty



.. math::

    \lambda_{} \|\beta\|_1


Let's form the problem

.. nbplot::

    >>> problem = rr.simple_problem(loss, penalty)
    >>> problem



.. math::

    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &= \ell^{\text{Gauss}}\left(X_{1}\beta\right) \\
    g(\beta) &= \lambda_{2} \|\beta\|_1 \\
    \end{aligned}



and solve it

.. nbplot::

    >>> beta = problem.solve(min_its=100)
    >>> beta
    array([  0.00000000e+00,  -0.00000000e+00,   4.34757960e+02,
             7.92364469e+01,   0.00000000e+00,   0.00000000e+00,
            -5.92308425e-11,   0.00000000e+00,   3.74915837e+02,
             0.00000000e+00])

Compare this to ``R``'s solution:

.. nbplot::

    >>> S = rpy2.r('diabetes_lars$beta[4,]')

Bound form
----------

We can also solve this in bound form

.. nbplot::

    >>> bound_form = rr.l1norm(p, bound=np.fabs(beta).sum())
    >>> bound_problem = rr.simple_problem(loss, bound_form)
    >>> bound_problem



.. math::

    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &= \ell^{\text{Gauss}}\left(X_{1}\beta\right) \\
    g(\beta) &= I^{\infty}(\|\beta\|_1 \leq \delta_{2}) \\
    \end{aligned}



Here is the solution

.. nbplot::

    >>> beta_bound = bound_problem.solve(min_its=100)
    >>> beta_bound
    array([ -0.00000000e+00,   0.00000000e+00,   4.34757960e+02,
             7.92364469e+01,  -0.00000000e+00,  -0.00000000e+00,
            -6.09077233e-11,  -0.00000000e+00,   3.74915837e+02,
            -0.00000000e+00])


