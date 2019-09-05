.. _lasso_example:

LASSO and variants
------------------

Here, we cover various optimization problems related to the LASSO.

.. nbplot::
   :format: python

    >>> import numpy as np
    >>> import regreg.api as rr
    >>> np.random.seed(0)
    >>> X = np.random.standard_normal((100, 10))
    >>> Y = np.random.standard_normal(100)

For a given :math:`X, Y`, here is the squared error loss

.. nbplot::
   :format: python

    >>> loss = rr.squared_error(X, Y)
    >>> loss #doctest: +SKIP


.. math::


   \frac{C}{2}\left\|X_{}\beta - Y_{}\right\|^2_2

The object ``loss`` is an instance of ``regreg.smooth.affine_smooth``
the representation of a smooth function in ``regreg`` composed with a
linear transformation. Its most important API piece is
``smooth_objective`` which evaluates the function, its gradient or both.

.. nbplot::
   :format: python

    >>> value, score_at_zero = loss.smooth_objective(np.zeros(loss.shape), 'both')
    >>> value #doctest: +SKIP
    41.482527733857864

.. nbplot::
   :format: python

    >>> score_at_zero, X.T.dot(X.dot(np.zeros(loss.shape)) - Y) #doctest: +SKIP
    (array([ -4.09173148,   2.70457691,   3.12078879, -32.7586658 ,
              9.43343624,   6.84661918, -12.55989746,   2.56336206,
            -11.58599406,  13.41032376]),
     array([ -4.09173148,   2.70457691,   3.12078879, -32.7586658 ,
              9.43343624,   6.84661918, -12.55989746,   2.56336206,
            -11.58599406,  13.41032376]))

The LASSO uses an :math:`\ell_1` penalty in "Lagrange" form:

.. math::


   \text{minimize}_{\beta} \frac{1}{2} \|Y-X\beta\|^2_2 + \lambda \|\beta\|_1.

.. nbplot::
   :format: python

    >>> penalty = rr.l1norm(10, lagrange=4.)
    >>> print('penalty:', str(penalty))  #doctest: +SKIP
    ('penalty:', 'l1norm((10,), lagrange=4.000000, offset=None)')

The object penalty is an instance of ``regreg.atoms.seminorm``. The main
API used in ``regreg`` is the method ``proximal`` which computes the
proximal mapping of the object. In ``regreg``, an ``atom`` generally
means it has a simple proximal map.

The proximal mapping of the function

.. math::

   f(\beta) = \lambda \|\beta\|_1

is

.. math::

   \text{prox}_{f, \epsilon}(z) = \text{argmin}_{\beta} \left[\frac{\epsilon}{2}\|\beta-z\|^2_2 + f(\beta)\right].

See `this
document <https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf>`__
for a brief review of proximal maps.

When :math:`f` is as above, this is the soft-thresholding map

.. math::


   \text{prox}_{f,\epsilon}(z)_i =
   \begin{cases}
   \text{sign}(z_i)(|z_i| - \lambda / \epsilon) & |z_i| > \lambda  / \epsilon \\
   0 & \text{otherwise.}
   \end{cases}

More generally, we might want to solve

.. math::

   \text{minimize}_{\beta} \left[\frac{C}{2} \|\beta-\mu\|^2_2 + \eta^T\beta + \gamma + f(\beta)\right]

which can easily done if we know the proximal mapping.

In ``regreg``, objects :math:`Q` of the form

.. math::


   Q(\beta) =  \frac{C}{2} \|\beta-\mu\|^2_2 + \eta^T\beta + \gamma

are represented instances of ``rr.identity_quadratic``.

.. nbplot::
   :format: python

    >>> Z = np.random.standard_normal(penalty.shape)
    >>> penalty.lagrange = 0.1
    >>> epsilon = 0.4
    >>> quadratic_term = rr.identity_quadratic(epsilon, Z, 0, 0)
    >>> penalty.proximal(quadratic_term) - penalty.solve(quadratic_term) #doctest: +SKIP
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

.. nbplot::
   :format: python

    >>> threshold = penalty.lagrange / epsilon
    >>> soft_thresh_Z = np.sign(Z) * (np.fabs(Z) - threshold) * (np.fabs(Z) > threshold)
    >>> soft_thresh_Z #doctest: +SKIP
    array([-0.8546166 ,  0.        ,  1.32886519, -0.54550055, -0.31643985,
           -0.05769128,  0.01902407,  0.27491786,  1.01741165,  0.24949823])

The objects ``loss`` and ``penalty`` are combined to form the LASSO
objective above. This is the canonical problem that we want to solve:

.. math::

   \text{minimize}_{\beta} f(\beta) + g(\beta)

where :math:`f` is a smooth convex function (i.e. we can compute its
value and its gradient) and :math:`g` is a function whose proximal map
is easy to compute.

The object ``rr.simple_problem`` requires its first argument to have a
``smooth_objective`` method and its second argument to have a ``solve``
method that solves

.. math::


   \text{minimize}_{\beta} g(\beta) + Q(\beta)

where :math:`Q` is a quadratic of the above form. If :math:`g` has a
``proximal`` method, this step just calls the proximal mapping.

.. nbplot::
   :format: python

    >>> penalty.lagrange = 4.
    >>> problem_lagrange = rr.simple_problem(loss, penalty)

Here is a mathematical representation of the problem computed using the :code:`_repr_latex_`
method of the problem.

.. math::
    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &= \frac{C}{2}\left\|X_{1}\beta - Y_{1}\right\|^2_2 \\
    g(\beta) &= \lambda_{2} \|\beta\|_1 \\
    \end{aligned}


.. nbplot::
   :format: python

    coef_lagrange = problem_lagrange.solve(tol=1.e-12)
    print('blah')
    print(coef_lagrange)

.. nbplot::
   :format: python

    >>> implied_bound = np.fabs(coef_lagrange).sum()
    >>> print(implied_bound) #doctest: +SKIP
    0.580132650282


.. nbplot::
   :format: python

    >>> bound_constraint = rr.l1norm(10, bound=implied_bound)
    >>> bound_constraint

.. math::

    I^{\infty}(\|\beta\|_1 \leq \delta_{})



.. nbplot::
   :format: python

    >>> problem_bound = rr.simple_problem(loss, bound_constraint)
    >>> problem_bound



.. math::

    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &= \frac{C}{2}\left\|X_{1}\beta - Y_{1}\right\|^2_2 \\
    g(\beta) &= I^{\infty}(\|\beta\|_1 \leq \delta_{2}) \\
    \end{aligned}



.. math::


   \begin{aligned}
   \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
   f(\beta) &= \frac{C}{2}\left\|X_{1}\beta - Y_{1}\right\|^2_2 \\
   g(\beta) &= I^{\infty}(\|\beta\|_1 \leq \delta_{2}) \\
   \end{aligned}

.. nbplot::
   :format: python

    >>> coef_bound = problem_bound.solve(tol=1.e-12)
    >>> print(coef_bound)

    [-0.          0.          0.          0.25644848 -0.09210037 -0.04856321
      0.07137744 -0.          0.04846725 -0.06317591]

[-0. 0. 0. 0.25644848 -0.09210037 -0.04856321
    0.07137744 -0. 0.04846725 -0.06317591]

.. nbplot::
   :format: python

    >>> np.linalg.norm(coef_bound - coef_lagrange) / np.linalg.norm(coef_lagrange)
    4.9118943989266597e-06

Comparison to ``sklearn``
-------------------------

The objective function is differs from ``sklearn.linear_model.Lasso`` by
a factor of :math:`1/n`.

.. nbplot::
   :format: python

    >>> from sklearn.linear_model import Lasso
    >>> clf = Lasso(alpha=penalty.lagrange / X.shape[0])
    >>> sklearn_soln = clf.fit(X, Y).coef_
    >>> sklearn_soln
    array([ 0.        , -0.        , -0.        ,  0.25887431, -0.08960121,
           -0.04907118,  0.07184117,  0.        ,  0.04895601, -0.06854384])

.. nbplot::
   :format: python

    >>> Xtiming = np.random.standard_normal((2000, 4000))
    >>> Ytiming = np.random.standard_normal(2000)
    >>> lagrange = np.fabs(Xtiming.T.dot(Ytiming)).max() * 0.6

.. nbplot::
   :format: python

    >>> clf = Lasso(alpha=lagrange / Xtiming.shape[0])
    >>> sklearn_soln = clf.fit(Xtiming, Ytiming).coef_

1 loop, best of 3: 296 ms per loop

.. nbplot::
   :format: python

    >>> loss_timing = rr.squared_error(Xtiming, Ytiming)
    >>> penalty_timing = rr.l1norm(Xtiming.shape[1], lagrange=lagrange)
    >>> coef_lagrange = rr.simple_problem(loss_timing, penalty_timing).solve(tol=1.e-12)

1 loop, best of 3: 257 ms per loop

.. nbplot::
   :format: python

    >>> loss_t = rr.squared_error(Xtiming, Ytiming)
    >>> penalty_t = rr.l1norm(Xtiming.shape[1], lagrange=lagrange)
    >>> soln1 = rr.simple_problem(loss_t, penalty_t).solve(tol=1.e-6)
    >>> clf = Lasso(alpha=lagrange / Xtiming.shape[0])
    >>> soln2 = clf.fit(Xtiming, Ytiming).coef_
    >>> print((soln1 != 0).sum(), (soln2 != 0).sum())
    >>> np.linalg.norm(soln1 - soln2) / np.linalg.norm(soln1)
    >>> (loss_t.smooth_objective(soln1, 'func') + np.fabs(soln1).sum() * lagrange, loss_t.smooth_objective(soln2, 'func') + np.fabs(soln2).sum() * lagrange)
    (965.57710749492287, 965.57847510945123)

    55 55


55 55

.. nbplot::
   :format: python

    >>> sklearn_soln
    array([ 0.,  0., -0., ...,  0., -0.,  0.])

.. nbplot::
   :format: python

    >>> np.linalg.norm(sklearn_soln - coef_lagrange) / np.linalg.norm(coef_lagrange)
    0.017723734004279082


Elastic net
===========

The elastic net differs from the LASSO only by addition of a quadratic
term. In ``regreg``, both smooth functions and atoms have their own
quadratic term that is added to the objective before solving the
problem.

The ``identity_quadratic`` is specified as :math:`Q` above:

.. math::

   Q(\beta) = \frac{C}{2} \|\beta-\mu\|^2_2 + \eta^T\beta + \gamma

with :math:`C` the first argument, :math:`\mu` the second, :math:`\eta`
the third and :math:`\gamma` the fourth.

.. nbplot::
   :format: python

    >>> enet_term = rr.identity_quadratic(0.5,0,0,0)
    >>> enet_term

.. math::

   \begin{equation*} \frac{L_{}}{2}\|\beta\|^2_2 \end{equation*}

.. nbplot::
   :format: python

    >>> penalty_enet = rr.l1norm(10, lagrange=4., quadratic=enet_term)
    >>> penalty_enet

.. math::

   \lambda_{} \|\beta\|_1 + \frac{L_{}}{2}\|\beta\|^2_2

.. nbplot::
   :format: python

    >>> problem_enet = rr.simple_problem(loss, penalty_enet)
    >>> enet_lagrange = problem_enet.solve(min_its=200, tol=1.e-12)
    >>> enet_lagrange
    array([ 0.        , -0.        , -0.        ,  0.25525639, -0.09145288,
           -0.04820132,  0.07108555,  0.        ,  0.04832992, -0.06302292])

Quadratic terms can also be added to problems as the first argument to
``solve``.

.. nbplot::
   :format: python

    >>> problem_lagrange.solve(enet_term, min_its=200, tol=1.e-12)
    array([ 0.        , -0.        , -0.        ,  0.25525639, -0.09145288,
           -0.04820132,  0.07108555,  0.        ,  0.04832992, -0.06302292])

Objects like ``enet_term`` are ubiquitous in ``regreg`` because it is a
package that uses proximal gradient methods to solve problems. Hence, it
is repeatedly solving problems like

.. math::

   \text{minimize}_{\beta} \frac{C}{2} \|z-\beta\|^2_2 + {\cal P}(\beta).

It therefore manipulates these objects in the course of solving the
problem. The arguments to ``rr.identity_quadratic`` determine functions
like

.. math::

   \beta \mapsto \frac{C}{2} \|\beta - \mu\|^2_2 + \beta^T\eta + \gamma.

.. nbplot::
   :format: python

    >>> C = 0.5
    >>> mu = np.arange(4)
    >>> eta = np.ones(4)
    >>> gamma = 2.3
    >>>
    >>> iq = rr.identity_quadratic(C, mu, eta, gamma)
    >>> str(iq)
    'identity_quadratic(0.500000, array([0, 1, 2, 3]), array([ 1.,  1.,  1.,  1.]), 2.300000)'

.. nbplot::
   :format: python

    >>> beta = -np.ones(4)
    >>> iq.objective(beta, 'func'), 0.5*C*((beta-mu)**2).sum() + (beta*eta).sum() + gamma
    (5.7999999999999998, 5.7999999999999998)

The arguments :math:`\mu` is the ``center`` and :math:`\eta` is the
``linear_term``, the argument :math:`\gamma` is ``constant`` which seems
somewhat unnecessary but is sometimes useful to track through
computations. such that ``center`` is 0.

.. nbplot::
   :format: python

    >>> str(iq.collapsed())
    'identity_quadratic(0.500000, 0.0, array([ 1. ,  0.5,  0. , -0.5]), 5.800000)'

As atoms and smooth functions have their own such quadratic terms, one
sometimes collects them to form an overall quadratic term

.. nbplot::
   :format: python

    >>> iq2 = rr.identity_quadratic(0.3, eta, mu, -2.1)
    >>> iq2



.. math::

    \begin{equation*} \frac{L_{}}{2}\|\beta-\mu_{}\|^2_2 + \left \langle \eta_{}, \beta \right \rangle + \gamma_{}  \end{equation*} 


.. nbplot::
   :format: python

    >>> str(iq+iq2)
    'identity_quadratic(0.800000, 0.0, array([ 0.7,  1.2,  1.7,  2.2]), 4.300000)'

.. nbplot::
   :format: python

    >>> iq.collapsed()

Dual problems
=============

The LASSO or Elastic Net can often be solved by solving an associated
dual problem. There are various ways to construct such problems.

One such way is to write our elastic net problem as

.. math::

   \text{minimize}_{\beta} f(\beta) + g(\beta)

where

.. math::

   \begin{aligned}
   f(\beta) &= \frac{1}{2} \|Y-X\beta\|^2_2 + \frac{C}{2} \|\beta\|^2_2 \\
   g(\beta) &= \lambda \|\beta\|_1.
   \end{aligned}

Then, we duplicate the variable :math:`\beta` yielding

.. math::

   \text{minimize}_{\beta_1,\beta_2:\beta_1=\beta_2} f(\beta_1) + g(\beta_2)

and introduce the Lagrangian

.. math::

   L(\beta_1,\beta_2,u) = f(\beta_1) + g(\beta_2) + u^T(\beta_1-\beta_2).

The dual problem is constructed by minimizing over
:math:`(\beta_1,\beta_2)` which yields a function of :math:`u`:

.. math::

   \inf_{\beta_1,\beta_2}L(\beta_1,\beta_2,u) = -f^*(-u) - g^*(u)

where

$$ f^\*(u) = \_{} ^Tu - f()

is the convex conjugate of :math:``f``. $$

The dual problem, written as a minimization problem is

.. math::

   \text{minimize}_{u} f^*(-u) + g^*(u).

In the elastic net case,

.. math::

   g^*(u) = I^{\infty}(\|u\|_{\infty} \leq \lambda)

and

.. math::

   \begin{aligned}
   f^*(-u) &= -\inf_{\beta}\left[ \frac{1}{2} \|Y-X\beta\|^2_2 + \frac{C}{2}\|\beta\|^2_2 + u^T\beta\right] \\
   \end{aligned}

We see the optimal :math:`\beta` in computing the infimum aboves
satisfies the normal equations

.. math::

   (X^TX + C \cdot I)\beta^*(u,Y) = X^TY - u

or

.. math::

   \beta^*(u,Y) = (X^TX+C \cdot I)^{-1}(X^TY-u).

Therefore,

.. math::

   f^*(-u) = \frac{1}{2} (X^TY-u)^T(X^TX+C \cdot I)^{-1}(X^TY-u) - \frac{1}{2}\|Y\|^2_2.

The function :math:`f^*` can be evaluated exactly as it is quadratic,
though it can also be solved numerically if our loss was not
squared-error. This is what the class ``regreg.api.conjugate`` does.

.. nbplot::
   :format: python

    >>> dual_loss = rr.conjugate(loss, negate=True, quadratic=enet_term, tol=1.e-12)
    >>> Q = np.linalg.inv(X.T.dot(X) + enet_term.coef * np.identity(10))
    >>>
    >>> def dual_loss_explicit(u):
    ...     z = X.T.dot(Y) - u
    ...     return 0.5 * (z * Q.dot(z)).sum() - 0.5 * (Y**2).sum()
    ...
    >>> U = np.random.standard_normal(10) * 1
    >>> print(np.linalg.norm((dual_loss.smooth_objective(U, 'grad') + Q.dot(X.T.dot(Y) - U)))  / np.linalg.norm(dual_loss.smooth_objective(U, 'grad')))
    1.88142740613e-06
    >>> print(dual_loss.smooth_objective(U, 'func'), dual_loss_explicit(U))
    (-33.914399021125028, -33.914399021125021)

.. nbplot::
   :format: python

    >>> dual_atom = penalty.conjugate
    >>> print(str(dual_atom))
    supnorm((10,), bound=4.000000, offset=None)



.. nbplot::
   :format: python

    >>> dual_problem = rr.simple_problem(dual_loss, dual_atom)
    >>> dual_soln = dual_problem.solve(min_its=50,tol=1.e-12)
    >>> dual_soln
    array([ 2.542112  , -0.91845618, -3.49499822,  4.        , -4.        ,
           -4.        ,  4.        ,  0.32799042,  4.        , -4.        ])

The solution to this dual problem is equal to the negative of the
gradient of the objective of our elastic net at the solution. This is
sometimes referred to as a primal-dual relationship, and is in effect a
restatement of the KKT conditions.

.. nbplot::
   :format: python

    >>> - loss.smooth_objective(enet_lagrange, 'grad') - enet_term.objective(enet_lagrange, 'grad')
    array([ 2.54211273, -0.91847349, -3.49504773,  4.        , -4.        ,
           -4.        ,  4.        ,  0.32793376,  4.        , -4.        ])

For the ``conjugate`` object, ``regreg`` retains a reference to the
minimizer, i.e. the gradient of the conjugate function. In our problem,
this is actually the solution to our elastic net problem, though it does
not have exact zeros.

.. nbplot::
   :format: python

    >>> primal_soln = dual_loss.argmin

.. nbplot::
   :format: python

    >>> primal_soln
    array([  1.52772852e-08,  -8.25787539e-07,  -9.38424377e-07,
             2.55256402e-01,  -9.14527151e-02,  -4.82011853e-02,
             7.10855973e-02,  -9.01763358e-08,   4.83299902e-02,
            -6.30227801e-02])

.. nbplot::
   :format: python

    >>> print(np.linalg.norm(primal_soln - enet_lagrange) / np.linalg.norm(enet_lagrange))
    4.33868837223e-06


We could alternatively have formed the explicit quadratic function for
:math:`f^*(-u)`. Having formed the quadratic objective explicitly, we
will have to also explicitly solve for the primal solution.

.. nbplot::
   :format: python

    >>> dual_quadratic = rr.quadratic_loss(Q.shape[0], Q=Q, offset=X.T.dot(Y))
    >>> dual_problem_alt = rr.simple_problem(dual_quadratic, dual_atom)
    >>> dual_soln_alt = dual_problem_alt.solve(min_its=100)
    >>> dual_soln_alt
    array([ 2.54211324, -0.9184747 , -3.49504942,  4.        , -4.        ,
           -4.        ,  4.        ,  0.32793405,  4.        , -4.        ])

.. nbplot::
   :format: python

    >>> primal_soln_alt = -dual_quadratic.smooth_objective(dual_soln_alt, 'grad')
    >>> print(np.linalg.norm(primal_soln_alt - enet_lagrange) / np.linalg.norm(enet_lagrange))

    9.57771586331e-08

Basis pursuit
=============

Yet another species in the zoology of LASSO problems is the basis
pursuit problem

.. math::

   \text{minimize}_{\beta: \|y-X\beta\|_2 \leq \delta} \|\beta\|_1.

This can be written as the sum of two atoms.

.. nbplot::
   :format: python

    >>> l1_part = rr.l1norm(X.shape[1], lagrange=1.)
    >>> l1_part


.. math::

    \lambda_{} \|\beta\|_1


.. nbplot::
   :format: python

    >>> X -= X.mean(0)[None,:]; Y -= Y.mean()
    >>> full_soln = np.linalg.pinv(X).dot(Y)
    >>> min_norm = np.linalg.norm(Y - X.dot(full_soln))
    >>> l2_part = rr.l2norm.affine(X, -Y, bound=1.1*min_norm) # we can't take a bound any smaller than sqrt(RSS)
    >>> l2_part



.. math::

    I^{\infty}(\|X_{}\beta - \alpha_{}\|_2 \leq \delta_{})


.. nbplot::
   :format: python

    >>> min_norm*1.1, np.linalg.norm(Y)
    (9.0308016267354709, 9.1084703609203732)

The problem can be turned into a problem solvable by ``regreg`` if we
smooth out ``l2_part``. This is related to the approaches taken by
``NESTA`` and ``TFOCS``.

There are quite a few variations, but one approach is to smooth the
``l2_part`` and solve a problem with a smoothed conjugate and an
:math:`\ell_1` penalty.

Smoothing out atoms
-------------------

.. nbplot::
   :format: python

    >>> small_q1 = rr.identity_quadratic(1.e-4, 0, 0, 0)
    >>> l2_part_smoothed = l2_part.smoothed(small_q1)
    >>> smoothed_problem = rr.simple_problem(l2_part_smoothed, l1_part)
    >>> smoothed_problem



.. math::

    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &=  \sup_{u \in \mathbb{R}^{p} } \left[ \langle X_{1}\beta, u \rangle - \left(\lambda_{1} \|u\|_2 + \frac{L_{1}}{2}\|u\|^2_2 + \left \langle \eta_{1}, u \right \rangle \right) \right] \\
    g(\beta) &= \lambda_{2} \|\beta\|_1 \\
    \end{aligned}



.. nbplot::
   :format: python

    >>> smoothed_soln = smoothed_problem.solve(min_its=10000)
    >>> smoothed_soln
    array([ 0.        , -0.        , -0.        ,  0.02233005, -0.        ,
           -0.        ,  0.        , -0.        ,  0.        , -0.        ])

TFOCS
=====

The TFOCS approach similarly smooths atoms, but solves this by adding a
small quadratic to the objective before solving a dual problem.
Formally, ``TFOCS`` solves a sequence of such smoothed problems where
the quadratic term is updated along the sequence. The center of the
quadratic is also updated along the sequence.

.. nbplot::
   :format: python

    >>> small_q2 = rr.identity_quadratic(1.e-6, 0, 0, 0)
    >>> l1_part2 = rr.l1norm(X.shape[1], lagrange=1., quadratic=small_q2)
    >>> linf_smoothed = l1_part2.conjugate
    >>> linf_smoothed



.. math::

     \sup_{u \in \mathbb{R}^{p} } \left[ \langle \beta, u \rangle - \left(\lambda_{} \|u\|_1 + \frac{L_{}}{2}\|u\|^2_2 \right) \right]


.. nbplot::
   :format: python

    >>> from regreg.affine import scalar_multiply, adjoint
    >>> transform, dual_atom = l2_part.dual
    >>> full_transform = adjoint(scalar_multiply(transform, -1))
    >>> tfocs_problem = rr.simple_problem(rr.affine_smooth(linf_smoothed, full_transform), dual_atom)
    >>> tfocs_problem



.. math::

    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &=  \sup_{u \in \mathbb{R}^{p} } \left[ \langle X_{1}\beta, u \rangle - \left(\lambda_{1} \|u\|_1 + \frac{L_{1}}{2}\|u\|^2_2 \right) \right] \\
    g(\beta) &= \lambda_{2} \|\beta\|_2 + \left \langle \eta_{2}, \beta \right \rangle \\
    \end{aligned}



.. nbplot::
   :format: python

    >>> tfocs_soln = tfocs_problem.solve(tol=1.e-12)

The primal solution is stored in the object ``linf_smoothed`` as
``grad`` which was the minimizer for the conjugate function before
applying ``full_transform``

.. nbplot::
   :format: python

    >>> primal_soln = linf_smoothed.grad
    >>> primal_soln
    array([ 0.        , -0.        , -0.        ,  0.02253928, -0.        ,
           -0.        ,  0.        , -0.        ,  0.        , -0.        ])

Elastic Net tutorial

The Elastic Net problem minimizes the objective

.. math::

   \frac{1}{2}||y - X\beta||^{2}_{2} + \lambda_{1}||\beta||_{1} + \lambda_2 \|\beta\|_2^2

To solve this problem using RegReg we begin by loading the necessary
numerical libraries

.. nbplot::
   :format: python

    >>> import numpy as np
    >>> import regreg.api as rr

Next, let's generate some example data,

.. nbplot::
   :format: python

    >>> X = np.random.normal(0,1,500000).reshape((500,1000))
    >>> Y = np.random.normal(0,1,500)

Now we can create the problem object, beginning with the loss function

.. nbplot::
   :format: python

    >>> loss = rr.quadratic_loss.affine(X,-Y, coef=0.5)
    >>> grouping = rr.quadratic_loss(1000, coef=1.)
    >>> sparsity = rr.l1norm(1000, lagrange=5.)

The penalty contains the regularization parameter that can be easily
accessed and changed,

.. nbplot::
   :format: python

    >>> grouping.coef
    >>> grouping.coef += 1
    >>> grouping.coef
    >>> sparsity.lagrange
    5.0

Now we can create the final problem object by comining the smooth
functions and the :math:``\\ell_1`` seminorm,

.. nbplot::
   :format: python

    >>> problem = rr.container(loss, grouping, sparsity)

The penalty parameters can still be changed by accessing grouping and
sparsity directly.

Next, we can select our algorithm of choice and use it solve the
problem,

.. nbplot::
   :format: python

    >>> solver = rr.FISTA(problem)
    >>> obj_vals = solver.fit(max_its=100, tol=1e-5)
    >>> solution = solver.composite.coefs

Here :code:`max_its` represents primal iterations, and tol is the primal
tolerance.

.. nbplot::
   :format: python

    >>> obj_vals
    array([ 133.36952356,   91.17516073,   82.88725763,   79.15453604,
             77.12832484,   75.90543802,   75.11101994,   74.63364029,
             74.34214746,   74.12245139,   73.96377444,   73.85787192,
             73.79442431,   73.75859708,   73.73718046,   73.72249764,
             73.71156917,   73.70358166,   73.69878815,   73.69596638,
             73.69419337,   73.69283107,   73.69166367,   73.69064932])

Basis pursuit
=============

In this tutorial, we demonstrate how to solve the basis pursuit problem
via a smoothing approach as in TFOCS. The basis pursuit problem is

.. math::


   \text{minimize}_{\beta: \|y-X\beta\| \leq \lambda} \|\beta\|_1

Let's generate some data first, setting the first 100 coefficients to be
large.

.. nbplot::
   :format: python

    >>> import regreg.api as R
    >>> import numpy as np
    >>> import scipy.linalg
    >>>
    >>> X = np.random.standard_normal((500,1000))
    >>>
    >>> beta = np.zeros(1000)
    >>> beta[:100] = 3 * np.sqrt(2 * np.log(1000))
    >>>
    >>> Y = np.random.standard_normal((500,)) + np.dot(X, beta)
    >>>
    >>> # Later, we will need this for a Lipschitz constant
    >>> Xnorm = scipy.linalg.eigvalsh(np.dot(X.T,X), eigvals=(998,999)).max()

The approach in TFOCS is to smooth the :math:`\ell_1` objective yielding
a dual problem

.. math::

   \text{minimize}_{u} \left(\|\beta\|_1 +
   \frac{\epsilon}{2} \|\beta\|^2_2 \right)^* \biggl|_{\beta=-X'u} + y'u + \lambda \|u\|_2

Above, :math:`f^*` denotes the convex conjugate. In this case, it is a
smoothed version of the unit :math:`\ell_{\infty}` ball constraint, as
its conjugate is the :math:`\ell_1` norm. Suppose we want to minimize
the :math:`\ell_1` norm achieving an explanation of 90% of the norm of
*Y*. That is,

.. math::


   \|Y - X\beta\|^2_2 \leq 0.1 \cdot \|Y\|^2_2

The code to construct the loss function looks like this

.. nbplot::
   :format: python

    >>> import regreg.api as R
    >>> linf_constraint = R.supnorm(1000, bound=1)
    >>> smoothq = R.identity_quadratic(0.01, 0, 0, 0)
    >>> smooth_linf_constraint = linf_constraint.smoothed(smoothq)
    >>> transform = R.linear_transform(-X.T)
    >>> loss = R.affine_smooth(smooth_linf_constraint, transform)
    >>> loss.quadratic = R.identity_quadratic(0, 0, Y, 0)
    >>> loss



.. math::

     \sup_{u \in \mathbb{R}^{p} } \left[ \langle X_{}\beta, u \rangle - \left(\lambda_{} \|u\|_1 + \frac{L_{}}{2}\|u\|^2_2 \right) \right] + \left \langle \eta_{}, \beta \right \rangle


The penalty is specified as

.. nbplot::
   :format: python

    >>> norm_Y = np.linalg.norm(Y)
    >>> l2_constraint_value = np.sqrt(0.1) * norm_Y
    >>> l2_lagrange = R.l2norm(500, lagrange=l2_constraint_value)

The container puts these together, then solves the problem by decreasing
the smoothing.

.. nbplot::
   :format: python

    >>> basis_pursuit_dual = R.simple_problem(loss, l2_lagrange)
    >>> basis_pursuit_dual


.. math::
    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &=  \sup_{u \in \mathbb{R}^{p} } \left[ \langle X_{1}\beta, u \rangle - \left(\lambda_{1} \|u\|_1 + \frac{L_{1}}{2}\|u\|^2_2 \right) \right] + \left \langle \eta_{1}, \beta \right \rangle \\
    g(\beta) &= \lambda_{2} \|\beta\|_2 \\
    \end{aligned}



.. nbplot::
   :format: python

    >>> solver = R.FISTA(basis_pursuit_dual)
    >>> tol = 1.0e-08
    >>>
    >>> for epsilon in [0.6**i for i in range(20)]:
    ...     smoothq = R.identity_quadratic(epsilon, 0, 0, 0)
    ...     smooth_linf_constraint = linf_constraint.smoothed(smoothq)
    ...     loss = R.affine_smooth(smooth_linf_constraint, transform)
    ...     basis_pursuit = R.simple_problem(loss, l2_lagrange)
    ...     solver = R.FISTA(basis_pursuit)
    ...     solver.composite.lipschitz = 1.1/epsilon * Xnorm
    ...     h = solver.fit(max_its=2000, tol=tol, min_its=10)
    ...
    >>> basis_pursuit_soln = smooth_linf_constraint.grad

The solution should explain about 90% of the norm of *Y*

.. nbplot::
   :format: python

    >>> print(1 - (np.linalg.norm(Y-np.dot(X, basis_pursuit_soln)) / norm_Y)**2)

    0.0

We now solve the corresponding bound form of the LASSO and verify we
obtain the same solution.

.. nbplot::
   :format: python

    >>> sparsity = R.l1norm(1000, bound=np.fabs(basis_pursuit_soln).sum())
    >>> loss = R.quadratic_loss.affine(X, -Y)
    >>> lasso = R.simple_problem(loss, sparsity)
    >>> lasso_solver = R.FISTA(lasso)
    >>> h = lasso_solver.fit(max_its=2000, tol=1.0e-10)
    >>> lasso_soln = lasso.coefs
    >>>
    >>> print(np.fabs(lasso_soln).sum(), np.fabs(basis_pursuit_soln).sum())
    >>> print(np.linalg.norm(Y-np.dot(X, lasso_soln)), np.linalg.norm(Y-np.dot(X, basis_pursuit_soln)))

    0.0 0.0
    2327.67420379 2327.67420379

.. nbplot::
   :format: python

    >>> import regreg.api as R
    >>> import numpy as np
    >>> import scipy.linalg
    >>> import pylab
    >>>
    >>> X = np.random.standard_normal((500,1000))
    >>> linf_constraint = R.supnorm(1000, bound=1)
    >>>
    >>> beta = np.zeros(1000)
    >>> beta[:100] = 3 * np.sqrt(2 * np.log(1000))
    >>>
    >>> Y = np.random.standard_normal((500,)) + np.dot(X, beta)
    >>> Xnorm = scipy.linalg.eigvalsh(np.dot(X.T,X), eigvals=(998,999)).max()
    >>>
    >>> smoothq = R.identity_quadratic(0.01, 0, 0, 0)
    >>> smooth_linf_constraint = linf_constraint.smoothed(smoothq)
    >>> transform = R.linear_transform(-X.T)
    >>> loss = R.affine_smooth(smooth_linf_constraint, transform)
    >>>
    >>> norm_Y = np.linalg.norm(Y)
    >>> l2_constraint_value = np.sqrt(0.1) * norm_Y
    >>> l2_lagrange = R.l2norm(500, lagrange=l2_constraint_value)
    >>>
    >>> basis_pursuit = R.simple_problem(loss, l2_lagrange)
    >>> solver = R.FISTA(basis_pursuit)
    >>> tol = 1.0e-08
    >>>
    >>> for epsilon in [0.6**i for i in range(20)]:
    ...    smoothq = R.identity_quadratic(epsilon, 0, 0, 0)
    ...    smooth_linf_constraint = linf_constraint.smoothed(smoothq)
    ...    loss = R.affine_smooth(smooth_linf_constraint, transform)
    ...    basis_pursuit = R.simple_problem(loss, l2_lagrange)
    ...    solver = R.FISTA(basis_pursuit)
    ...    solver.composite.lipschitz = 1.1/epsilon * Xnorm
    ...    h = solver.fit(max_its=2000, tol=tol, min_its=10)
    ...
    >>> basis_pursuit_soln = smooth_linf_constraint.grad
    >>>
    >>> sparsity = R.l1norm(1000, bound=np.fabs(basis_pursuit_soln).sum())
    >>> loss = R.quadratic_loss.affine(X, -Y)
    >>> lasso = R.container(loss, sparsity)
    >>> lasso_solver = R.FISTA(lasso)
    >>> lasso_solver.fit(max_its=2000, tol=1.0e-10)
    >>> lasso_soln = lasso.coefs
    >>>
    >>> pylab.plot(basis_pursuit_soln, label='Basis pursuit')
    >>> pylab.plot(lasso_soln, label='LASSO')
    >>> pylab.legend()
    <...>


