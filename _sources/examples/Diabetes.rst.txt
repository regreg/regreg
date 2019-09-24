.. _diabetes_example:


The Diabetes data from LARS
---------------------------

.. nbplot::
    :format: python

    import numpy as np, regreg.api as rr
    import rpy2.robjects as rpy2

Let's grab the diabetes data from the lars package in R

.. nbplot::
    :format: python

    rpy2.r('''
    suppressMessages(library(lars))
    data(diabetes)
    X = diabetes$x
    Y = diabetes$y
    diabetes_lars = lars(diabetes$x, diabetes$y, type='lasso')
    L = diabetes_lars$lambda
    ''')
    X = rpy2.r('X')
    L = rpy2.r('L')
    Y = rpy2.r('Y')

.. nbplot::
    :format: python

    X = np.asarray(X)
    Y = np.asarray(Y)
    n, p = X.shape
    n, p

Our loss function and penalty

.. nbplot::
    :format: python

    loss = rr.glm.gaussian(X, Y)
    loss



.. math::

    \ell^{\text{Gauss}}\left(X_{}\beta\right)


.. math::


   \ell^{\text{Gauss}}\left(X_{}\beta\right)

Now, our penalty:

.. nbplot::
    :format: python

    penalty = rr.l1norm(X.shape[1], lagrange=L[3])
    penalty



.. math::

    \lambda_{} \|\beta\|_1


.. math::


   \lambda_{} \|\beta\|_1

Let's form the problem

.. nbplot::
    :format: python

    problem = rr.simple_problem(loss, penalty)
    problem



.. math::

    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &= \ell^{\text{Gauss}}\left(X_{1}\beta\right) \\
    g(\beta) &= \lambda_{2} \|\beta\|_1 \\
    \end{aligned}



.. math::


   \begin{aligned}
   \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
   f(\beta) &= \ell^{\text{Gauss}}\left(X_{1}\beta\right) \\
   g(\beta) &= \lambda_{2} \|\beta\|_1 \\
   \end{aligned}

and solve it

.. nbplot::
    :format: python

    beta = problem.solve(min_its=100)
    beta

Compare this to ``R``'s solution:

.. nbplot::
    :format: python

    S = rpy2.r('diabetes_lars$beta[4,]')
    np.asarray(S)

Bound form
==========

We can also solve this in bound form

.. nbplot::
    :format: python

    bound_form = rr.l1norm(p, bound=np.fabs(beta).sum())
    bound_problem = rr.simple_problem(loss, bound_form)
    bound_problem



.. math::

    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &= \ell^{\text{Gauss}}\left(X_{1}\beta\right) \\
    g(\beta) &= I^{\infty}(\|\beta\|_1 \leq \delta_{2}) \\
    \end{aligned}



.. math::

   \begin{aligned}
   \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
   f(\beta) &= \ell^{\text{Gauss}}\left(X_{1}\beta\right) \\
   g(\beta) &= I^{\infty}(\|\beta\|_1 \leq \delta_{2}) \\
   \end{aligned}

Here is the solution

.. nbplot::
    :format: python

    beta_bound = bound_problem.solve(min_its=100)
    beta_bound

.. code-links::
   :timeout: -1

