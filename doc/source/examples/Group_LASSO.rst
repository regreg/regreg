
The group LASSO for least squares
=================================

This notebook provides some ways to solve the group LASSO problem

.. math::


   \frac{1}{2} \|Y-X\beta\|^2_2 + \lambda \sum_i \|\beta[g_i]\|_2

where :math:`g_i` are pairwise distinct subsets of
:math:`\{1, \dots, p\}`.

.. nbplot::

    >>> # third party imports
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import rpy2.robjects as rpy2
    >>> from rpy2.robjects import numpy2ri
    >>> numpy2ri.activate()
    >>>
    >>> # the regreg import
    >>> import regreg.api as rr

Comparison to gglasso on small data
-----------------------------------

We will compare to the ``gglasso`` packge in ``R`` using the dataset
supplied by this package. We see that for smaller problems like this,
``gglasso`` is faster, particularly if we use fairly high tolerance for
convergence.

.. nbplot::

    >>> rpy2.r('''
    >>> library(gglasso)
    >>> data(bardet)
    >>> group1 = rep(1:20,each=5)
    >>> Y = bardet$y
    >>> X = bardet$x 
    >>> Xr = matrix(rnorm(nrow(X)*ncol(X)), nrow(X), ncol(X))
    >>> gg1 = gglasso(x=X,y=Y,group=group1,loss="ls")
    >>> B = gg1$beta
    >>> L = gg1$lambda
    >>> Lr = gglasso(x=Xr,y=Y,group=group1,loss="ls")$lambda
    >>> DF = gg1$df
    >>> ''')
    >>> B, DF, L, Y, X, Xr, Lr = [rpy2.r(name) for name in ('B', 'DF', 'L', 'Y', 'X', 'Xr', 'Lr')]

The ``gglasso`` centers ``X`` and ``Y`` by default and its loss is

.. math::


   \frac{1}{2n} \|Y-X\beta\|^2_2

.. nbplot::

    >>> Y -= Y.mean()
    >>> X -= X.mean(0)[np.newaxis,:]
    >>> n, p = X.shape
    >>> loss = rr.squared_error(X, Y, coef=1./n)
    >>> loss



.. math::

    \frac{C}{2}\left\|X_{}\beta - Y_{}\right\|^2_2


The groups used in the example are all of size 5.

.. nbplot::

    >>> groups = []
    >>> for i in range(20):
    ...     groups.extend([i]*5)
    >>> penalty = rr.group_lasso(groups, lagrange=L.max())
    >>> penalty



.. math::

    \lambda_{} \sum_g w_g \|\beta[g]\|_2


It is sometimes useful to have a global Lipschitz constant to start the
algorithm so the backtracking step does not have a long search. For this
loss we can take the largest eigenvalue of :math:`X^TX/n`:

.. math::


   \|\nabla {\mathcal L}\|_{\mathrm{Lip}} = \frac{1}{n}\|X\|_{\mathrm{op}}^2

.. nbplot::

    >>> lipschitz = rr.power_L(X)**2 / n

We are all set to specify the problem and solve it. This is a simple
problem in that its proximal operator is separable. It can be specified
with the ``simple_problem`` class.

.. nbplot::

    >>> problem = rr.simple_problem(loss, penalty)
    >>> problem



.. math::

    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &= \frac{C}{2}\left\|X_{1}\beta - Y_{1}\right\|^2_2 \\
    g(\beta) &= \lambda_{2} \sum_g w_g \|\beta[g]\|_2 \\
    \end{aligned}



With this choice of ``lagrange`` parameter the solution should be 0.

.. nbplot::

    >>> coefs = problem.solve()
    >>> (coefs != 0).sum()
    5

The problem could also be solved by a straightforward generalized
gradient algorithm that does no backtracking. This generally does not
work as well because the global Lipschitz constant is much larger than
it has to be. This generalized gradient algorithm can be found in
``regreg.problems.simple.gengrad``.

Constructing a path of solutions (not using strong rules)
---------------------------------------------------------

The package ``gglasso`` chooses which penalty parameters to use as
follows:

.. nbplot::

    >>> score0 = loss.smooth_objective(np.zeros(loss.shape), mode='grad')
    >>> dual_penalty = rr.group_lasso_dual(groups, lagrange=1.)
    >>> lagrange_max = dual_penalty.seminorm(score0)
    >>> lagrange_seq = lagrange_max * np.exp(np.linspace(np.log(0.001), 0, 100))[::-1]
    >>> np.linalg.norm(L - lagrange_seq)
    1.0640364669290794e-17

Let's write a function that solves the group LASSO for a grid of
:math:`\lambda` values.

.. nbplot::

    >>> def solve_path(X, Y, groups, lagrange_seq, tol=1.e-8, max_its=50):
    ...
    ...     lagrange_seq = np.sort(lagrange_seq)[::-1]
    ...     loss = rr.squared_error(X, Y, coef=1./n)
    ...     penalty = rr.group_lasso(groups, lagrange=lagrange_seq.max())
    ...     problem = rr.simple_problem(loss, penalty)
    ...     solns = [problem.solve(tol=tol, min_its=20, max_its=max_its)]
    ...     final_step = problem.final_step
    ...     for lagrange in lagrange_seq[1:]:
    ...         penalty.lagrange = lagrange
    ...         solns.append(problem.solve(start_step=final_step, tol=tol, max_its=max_its).copy())
    ...         final_step = problem.final_step
    ...     return np.array(solns), problem

.. nbplot::

    >>> # %timeit solve_path(X, Y, groups, L)

.. nbplot::

    >>> # %%timeit
    >>> _ = rpy2.r('G=gglasso(x=X,y=Y,group=group1,loss="ls");')

We see that ``gglasso`` is much faster for this design, though we'll see
that the objective values are not quite as low as ``regreg``.

Let's compare the solutions to see they are at least similar. Below, we
will see that ``gglasso``'s advantage diminishes in larger problems.
This suggests that at least part of this time is simply the time needed
to call the appropriate methods for ``regreg`` which solves generic
problems rather than the one that ``gglasso`` is specialized to solve.

Another part of the problem is that ``regreg`` is not taking advantages
of the so-called `strong rules <strong%20rules%20paper>`__ when solving
along the path.

.. nbplot::

    >>> plt.figure(figsize=(6,6))
    >>> solns = solve_path(X, Y, groups, L)[0]
    >>> [plt.plot(np.log(L), solns[:,i]) for i in range(100)];

.. nbplot::

    >>> plt.figure(figsize=(6,6))
    >>> [plt.plot(np.log(L), B[i]) for i in range(100)];

To solve the problem half waydown the path, ``regreg`` is faster. We
could also use the `strong rules <strong%20rules%20paper>`__ to speed up
``regreg`` as it is actually doing a full for each value of
:math:`\lambda`.

.. nbplot::

    >>> def solve_one(X, Y, groups, lagrange, tol=1.e-8, max_its=100):
    ...     loss = rr.squared_error(X, Y, coef=1./n)
    ...     penalty = rr.group_lasso(groups, lagrange=lagrange)
    ...     problem = rr.simple_problem(loss, penalty)
    ...     return problem.solve(tol=tol, max_its=max_its).copy(), problem

.. nbplot::

    >>> L_test = L[int(len(L)/2)-1]
    >>> # %timeit solve_one(X, Y, groups, L_test)

Let's see how ``gglasso`` does to get at the same point. To be fair, we
will only take 50 steps to get there.

.. nbplot::

    >>> rpy2.r.assign('L_test', L_test)
    >>> _ = rpy2.r('L_half = exp(seq(log(max(L)), log(L_test), length=50))')

.. nbplot::

    >>> # %%timeit
    >>> _ = rpy2.r('gglasso(x=X,y=Y,group=group1,loss="ls",lambda=L_half)')

Let's compare objective values.

.. nbplot::

    >>> B_mid = rpy2.r('gglasso(x=X,y=Y,group=group1,loss="ls",lambda=L_half)$beta[,50]')
    >>> soln, problem = solve_one(X, Y, groups, L_test)
    >>> problem.objective(soln), problem.objective(B_mid)
    (0.003015701266213722, 0.0030206831554445077)

If we relax the tolerance a bit, ``regreg`` is even faster.

.. nbplot::

    >>> # %timeit solve_one(X, Y, groups, L_test, tol=1.e-7)

But, its objective value is still a little worse than before, though
still better than ``gglasso``.

.. nbplot::

    >>> soln, problem = solve_one(X, Y, groups, L_test, tol=1.e-7)
    >>> problem.objective(soln), problem.objective(B_mid)
    (0.0030165785350338477, 0.0030206831554445077)

Comparison on a random design
-----------------------------

Let's see how they compare on a random design, perhaps this design is
particularly fast for the coordinate descent method.

.. nbplot::

    >>> rpy2.r('''
    >>> Xr = matrix(rnorm(nrow(X)*ncol(X)), nrow(X), ncol(X))
    >>> Lr = gglasso(x=Xr,y=Y,group=group1,loss="ls")$lambda
    >>> ''')
    >>> Lr = rpy2.r('Lr')
    >>> Xr = rpy2.r('Xr')

.. nbplot::

    >>> # %timeit rpy2.r('gglasso(x=Xr,y=Y,group=group1,loss="ls")')
    >>> # %timeit solve_path(Xr, Y, groups, Lr)

Comparison of objective values.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's compare the objective values. They are very close.

.. nbplot::

    >>> loss = rr.squared_error(X, Y, coef=1./n)
    >>> penalty = rr.group_lasso(groups, lagrange=L.max())
    >>> problem = rr.simple_problem(loss, penalty)
    >>> plt.figure(figsize=(6,6))
    >>> obj_vals = []
    >>> for i, lagrange in enumerate(L):
    ...     penalty.lagrange = lagrange
    ...     val1 = problem.objective(B[:,i])
    ...     val2 = problem.objective(solns[i])
    ...     obj_vals.append((val1, val2, val1 - val2))
    >>> obj_vals = np.array(obj_vals)
    >>> plt.plot(np.log(L), obj_vals[:,0], label='gglasso')
    >>> plt.plot(np.log(L), obj_vals[:,1], label='regreg')
    >>> plt.legend(loc='lower right')
    <...>

For smaller values of the regularization parameter, ``regreg`` reaches a
lower objective value, though the difference is fairly small.

.. nbplot::

    >>> plt.figure(figsize=(6,6))
    >>> plt.plot(np.log(L), obj_vals[:,2])
    [...]

Larger problems
---------------

Let's generate some larger data and time their performance.

.. nbplot::

    >>> n, p, s =  1000, 10000, 200
    >>> Xb = np.random.standard_normal((n, p)) / np.sqrt(n)
    >>> beta = np.zeros(p); beta[:s] = 6.
    >>> np.random.shuffle(beta)
    >>> Yb = Xb.dot(beta) + np.random.standard_normal(n)
    >>> groupsb = []
    >>> for i in range(20):
    ...     groupsb.extend([i+1]*int(p/20))
    >>> groups = np.array(groupsb)
    >>> rpy2.r.assign('Xb', Xb)
    >>> rpy2.r.assign('Yb', Yb)
    >>> _ = rpy2.r.assign('groupsb', groups)

.. nbplot::

    >>> # %%timeit
    >>> rpy2.r('Lb = gglasso(x=Xb,y=Yb,group=groupsb,loss="ls")$lambda')
    >>> Lb = rpy2.r('Lb')

.. nbplot::

    >>> Yb -= Yb.mean()
    >>> Xb -= Xb.mean(0)[np.newaxis,:]

.. nbplot::

    >>> # %timeit solve_path(Xb, Yb, groupsb, Lb)

.. nbplot::

    >>> Lb_test = Lb[int(len(Lb)/2)]
    >>> rpy2.r.assign('Lb_test', Lb_test)
    >>> rpy2.r('Lb_half = exp(seq(log(max(Lb)), log(Lb_test), length=50));')
    >>> # %timeit solve_one(Xb, Yb, groupsb, Lb_test, tol=1.e-10, max_its=200)
    >>> solnb, problemb = solve_one(Xb, Yb, groupsb, Lb_test, tol=1.e-10, max_its=150)

.. nbplot::

    >>> # %%timeit
    >>> _ = rpy2.r('gglasso(x=Xb,y=Yb,group=groupsb,loss="ls",lambda=Lb_half)')

.. nbplot::

    >>> Bb_mid = rpy2.r('gglasso(x=Xb, y=Yb, group=groupsb, loss="ls", lambda=Lb_half)$beta[,50]')

.. nbplot::

    >>> problemb.objective(solnb), problemb.objective(Bb_mid)
    (1.7081065654113237, 1.7081416103445459)

Comparison of objective values
------------------------------

.. nbplot::

    >>> Bb = rpy2.r('gglasso(x=Xb, y=Yb, group=groupsb, loss="ls")$beta')

.. nbplot::

    >>> solns, problemb = solve_path(Xb, Yb, groups, Lb)

.. nbplot::

    >>> plt.figure(figsize=(6,6))
    >>> obj_vals = []
    >>> for i, lagrange in enumerate(L):
    ...     penalty.lagrange = lagrange
    ...     val1 = problemb.objective(Bb[:,i])
    ...     val2 = problemb.objective(solns[i])
    ...     obj_vals.append((val1, val2, val1 - val2))
    >>> obj_vals = np.array(obj_vals)
    >>> plt.plot(np.log(L), obj_vals[:,0], label='gglasso')
    >>> plt.plot(np.log(L), obj_vals[:,1], label='regreg')
    >>> plt.legend()
    <...>

.. nbplot::

    >>> plt.figure(figsize=(6,6))
    >>> plt.plot(obj_vals[:,2])
    [...]

.. nbplot::

    >>> numpy2ri.deactivate()


