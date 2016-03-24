.. _mixedtutorial:

Mixing seminorms tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial illustrates how to use RegReg to solve problems that have seminorms in both the objective and the contraint. We illustrate with an example:

.. math::

       \frac{1}{2}||y - \beta||^{2}_{2} + \lambda \|\beta\|_1 \text{ subject to} \  ||D\beta||_{1} \leq \delta   

with

.. math::

       D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)

To solve this problem using RegReg we begin by loading the necessary numerical libraries

.. ipython::

   import numpy as np
   from scipy import sparse
   import regreg.api as rr

Next, let's generate an example signal, and solve the Lagrange form of the problem

.. ipython::
 
   Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
   loss = rr.quadratic.shift(-Y, coef=0.5)

   sparsity = rr.l1norm(len(Y), lagrange=1.4)
   # TODO should make a module to compute typical Ds
   D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
   fused = rr.l1norm.linear(D, lagrange=25.5)
   problem = rr.container(loss, sparsity, fused)
   
   solver = rr.FISTA(problem)
   solver.fit(max_its=100, tol=1e-10)
   solution = solver.composite.coefs

We will now solve this problem in constraint form, using the 
achieved  value :math:`\delta = \|D\widehat{\beta}\|_1`.
By default, the container class will try to solve this problem with the two-loop strategy.

.. ipython::

   delta = np.fabs(D * solution).sum()
   sparsity = rr.l1norm(len(Y), lagrange=1.4)
   fused_constraint = rr.l1norm.linear(D, bound=delta)
   constrained_problem = rr.container(loss, fused_constraint, sparsity)
   constrained_solver = rr.FISTA(constrained_problem)
   constrained_solver.composite.lipschitz = 1.01
   constrained_solver.perform_backtrack = False
   vals = constrained_solver.fit(max_its=10, tol=1e-06, monotonicity_restart=False)
   constrained_solution = constrained_solver.composite.coefs


We can now check that the obtained value matches the constraint,

.. ipython::

   constrained_delta = np.fabs(D * constrained_solution).sum()
   print delta, constrained_delta
   print np.linalg.norm(solution - constrained_solution) / np.linalg.norm(solution)


.. plot::

    import numpy as np
    import pylab	
    from scipy import sparse
    import regreg.api as R

    Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
    loss = R.quadratic.shift(-Y, coef=0.5)

    sparsity = R.l1norm(len(Y), lagrange=1.4)
    # TODO should make a module to compute typical Ds
    D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
    fused = R.l1norm.linear(D, lagrange=25.5)
    problem = R.container(loss, sparsity, fused)

    solver = R.FISTA(problem)
    solver.fit(max_its=100, tol=1e-10)
    solution = solver.composite.coefs

    delta = np.fabs(D * solution).sum()
    sparsity = R.l1norm(len(Y), lagrange=1.4)
    fused_constraint = R.l1norm.linear(D, bound=delta)
    constrained_problem = R.container(loss, fused_constraint, sparsity)
    constrained_solver = R.FISTA(constrained_problem)
    constrained_solver.composite.lipschitz = 1.01
    constrained_solver.perform_backtrack = False
    vals = constrained_solver.fit(max_its=10, tol=1e-06, monotonicity_restart=False)
    constrained_solution = constrained_solver.composite.coefs

    constrained_delta = np.fabs(D * constrained_solution).sum()
    print delta, constrained_delta

    pylab.scatter(np.arange(Y.shape[0]), Y)
    pylab.plot(solution, c='y', linewidth=3)	
    pylab.plot(constrained_solution, c='r', linewidth=1)
    #pylab.plot(conjugate_coefs, c='black', linewidth=3)	
    #pylab.plot(conjugate_coefs_gen, c='gray', linewidth=1)		

