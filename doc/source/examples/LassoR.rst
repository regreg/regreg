
.. nbplot::

    >>> import numpy as np, regreg.api as rr
    >>> import rpy2.robjects as rpy2

The Diabetes data from LARS
---------------------------

.. nbplot::

    >>> rpy2.r('''
    >>> install.packages('lars',repos='http://cloud.r-project.org')
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

    /home/jb/virtualenvs/regreg/lib/python3.5/site-packages/rpy2/rinterface/__init__.py:186: RRuntimeWarning: Installing package into ‘/home/jb/R/x86_64-pc-linux-gnu-library/3.2’
    (as ‘lib’ is unspecified)
    
      warnings.warn(x, RRuntimeWarning)
    /home/jb/virtualenvs/regreg/lib/python3.5/site-packages/rpy2/rinterface/__init__.py:186: RRuntimeWarning: trying URL 'http://cloud.r-project.org/src/contrib/lars_1.2.tar.gz'
    
      warnings.warn(x, RRuntimeWarning)
    /home/jb/virtualenvs/regreg/lib/python3.5/site-packages/rpy2/rinterface/__init__.py:186: RRuntimeWarning: Content type 'application/x-gzip'
      warnings.warn(x, RRuntimeWarning)
    /home/jb/virtualenvs/regreg/lib/python3.5/site-packages/rpy2/rinterface/__init__.py:186: RRuntimeWarning:  length 173620 bytes (169 KB)
    
      warnings.warn(x, RRuntimeWarning)
    /home/jb/virtualenvs/regreg/lib/python3.5/site-packages/rpy2/rinterface/__init__.py:186: RRuntimeWarning: =
      warnings.warn(x, RRuntimeWarning)
    /home/jb/virtualenvs/regreg/lib/python3.5/site-packages/rpy2/rinterface/__init__.py:186: RRuntimeWarning: 
    
      warnings.warn(x, RRuntimeWarning)
    /home/jb/virtualenvs/regreg/lib/python3.5/site-packages/rpy2/rinterface/__init__.py:186: RRuntimeWarning: downloaded 169 KB
    
    
      warnings.warn(x, RRuntimeWarning)

##STDOUT_END####STDOUT_START##
    
    
    
    The downloaded source packages are in
    	‘/tmp/RtmpEEcaLO/downloaded_packages’
    
    
    

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


