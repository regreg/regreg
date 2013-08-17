In[1]:

.. ipython::

    import numpy as np, regreg.api as rr
    %pylab inline
    %load_ext rmagic

.. parsed-literal::

    
    Welcome to pylab, a matplotlib-based Python environment [backend: module://Ipython::.kernel.zmq.pylab.backend_inline].
    For more information, type 'help(pylab)'.


In[2]:

.. ipython::

    %%R -o X,L,Y
    library(lars)
    data(diabetes)
    X = diabetes$x
    Y = diabetes$y
    diabetes_lars = lars(diabetes$x, diabetes$y, type='lasso')
    L = diabetes_lars$lambda

.. parsed-literal::

    Loaded lars 1.1
    


In[3]:

.. ipython::

    n, p = X.shape

In[4]:

.. ipython::

    loss = rr.squared_error(X,Y)
    Y = Y - Y.mean()
    penalty = rr.l1norm(X.shape[1], lagrange=L[3])
    penalty.lagrange

Out[4]:

.. parsed-literal::

    316.07405269831145

In[5]:

.. ipython::

    problem = rr.simple_problem(loss, penalty)
    beta = problem.solve(min_its=100)
    beta

Out[5]:

.. parsed-literal::

    array([   0.        ,   -0.        ,  434.75795962,   79.23644688,
              0.        ,    0.        ,   -0.        ,    0.        ,
            374.91583685,    0.        ])

In[6]:

.. ipython::

    %%R -o S
    S = diabetes_lars$beta[4,]

In[7]:

.. ipython::

    S

Out[7]:

.. parsed-literal::

    array([   0.        ,    0.        ,  434.75795962,   79.23644688,
              0.        ,    0.        ,    0.        ,    0.        ,
            374.91583685,    0.        ])

In[8]:

.. ipython::

    bound_form = rr.l1norm(p, bound=np.fabs(beta).sum())
    bound_problem = rr.simple_problem(loss, bound_form)
    beta_bound = bound_problem.solve(min_its=100)
    beta_bound

Out[8]:

.. parsed-literal::

    array([  -0.        ,    0.        ,  434.75795962,   79.23644688,
             -0.        ,   -0.        ,   -0.        ,   -0.        ,
            374.91583685,   -0.        ])

In[17]:

.. ipython::

    bound_value = np.linalg.norm(Y - np.dot(X, beta))
    basis_pursuit_constraint = rr.l2norm.affine(X, -Y, bound=bound_value)
    penalty.lagrange
    
    smoothed = basis_pursuit_constraint.smoothed(rr.identity_quadratic(1.e-5,0,0,0))
    smoothed_problem = rr.simple_problem(smoothed, penalty)
    smoothed_problem.solve(min_its=100)

Out[17]:

.. parsed-literal::

    array([   0.        ,   -0.        ,  434.75407411,   79.18744169,
              0.        ,    0.        ,   -0.        ,    0.        ,
            374.92041835,    0.        ])

In[18]:

.. ipython::

    smoothed.grad
    import pylab
    pylab.scatter(smoothed.grad, Y - np.dot(X, beta))
    pylab.show()

.. image:: LASSO_again_files/LASSO_again_fig_00.png

In[11]:

.. ipython::

    transform, dual = basis_pursuit_constraint.dual
    P, D = rr.tfocs(penalty, transform, dual, epsilon=[1]*50 + [0.01]*10)

In[19]:

.. ipython::

    plt.scatter(D, smoothed.grad)

Out[19]:

.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x1103601d0>

.. image:: LASSO_again_files/LASSO_again_fig_01.png


