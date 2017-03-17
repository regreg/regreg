
Huberized lasso tutorial
========================

The Huberized lasso minimizes the following objective

.. math::


   H_{\delta}(Y - X\beta) + \lambda \|\beta\|_1

 where :math:`H_{\delta}(\cdot)` is a function applied element-wise,

.. math::


    H_{\delta}(r) = \left\{\begin{array}{ll} r^2/2 & \mbox{ if } |r| \leq \delta \\ \delta r - \delta^2/2 & \mbox{ else}\end{array} \right.

 To solve this problem using RegReg we begin by loading the necessary
numerical libraries

.. nbplot::

    >>> import numpy as np
    >>> import regreg.api as rr

    /home/jb/code/regreg/regreg/smooth/glm.py:1152: UserWarning: unable to import PHReg from statsmodels, objective function is the zero function!
      warnings.warn('unable to import PHReg from statsmodels, objective function is the zero function!')

Next, let's generate some example data,

.. nbplot::

    >>> X = np.random.normal(0,1,500000).reshape((500,1000))
    >>> Y = np.random.randint(0,2,500)

Now we can create the problem object, beginning with the loss function

.. nbplot::

    >>> penalty = rr.l1norm(1000,lagrange=5.)
    >>> loss = rr.l1norm.affine(X,-Y, lagrange=1.).smoothed(rr.identity_quadratic(1,0,0,0))

The penalty contains the regularization parameter that can be easily
accessed and changed,

.. nbplot::

    >>> penalty.lagrange
    5.0

Now we can create the final problem object

.. nbplot::

    >>> problem = rr.simple_problem(loss, penalty)

Next, we can select our algorithm of choice and use it solve the
problem,

.. nbplot::

    >>> solver = rr.FISTA(problem)
    >>> obj_vals = solver.fit(max_its=200, tol=1e-6)
    >>> solution = solver.composite.coefs
    >>> (solution != 0).sum()
    362

Here max\_its represents primal iterations, and tol is the primal
tolerance.

.. nbplot::

    >>> obj_vals
    array([ 126.        ,   75.78267624,   66.8357571 ,   62.72644628,
             60.32605584,   58.77165373,   57.78010601,   57.08972094,
             56.60401379,   56.24194582,   55.9622382 ,   55.74287594,
             55.57336482,   55.44781422,   55.34988045,   55.27196206,
             55.21011602,   55.1640617 ,   55.13135902,   55.10969164,
             55.09394259,   55.08135339,   55.07153922,   55.06414123,
             55.05863748,   55.05514647,   55.05288021,   55.05150806,
             55.05058692,   55.04974983,   55.04884254,   55.04781529,
             55.04671874,   55.04561987,   55.04458707,   55.04367244,
             55.0429083 ,   55.04230126,   55.04182898,   55.04146411,
             55.04120032,   55.04099191,   55.04081546,   55.04066194,
             55.04052909,   55.04041589,   55.04032385])

Poisson regression tutorial
===========================

The Poisson regression problem minimizes the objective

.. math::


   -2 \left(Y^TX\beta - \sum_{i=1}^n \mbox{exp}(x_i^T\beta) \right), \qquad Y_i \in {0,1,2,\ldots}

 which corresponds to the usual Poisson regression model

.. math::


   P(Y_i=j) = \frac{\mbox{exp}(jx_i^T\beta-\mbox{exp}(x_i^T\beta))}{j!}

To solve this problem using RegReg we begin by loading the necessary
numerical libraries

.. nbplot::

    >>> import numpy as np
    >>> import regreg.api as rr

The only code needed to add Poisson regression is a class with one
method which computes the objective and its gradient.

Next, let's generate some example data,

.. nbplot::

    >>> n = 1000
    >>> p = 50
    >>> X = np.random.standard_normal((n,p))
    >>> Y = np.random.randint(0,100,n)

Now we can create the problem object, beginning with the loss function

.. nbplot::

    >>> loss = rr.glm.poisson(X, counts=Y)

    /home/jb/code/regreg/regreg/smooth/glm.py:836: RuntimeWarning: divide by zero encountered in log
      loss_terms = - coef * ((counts - 1) * np.log(counts))

Next, we can fit this model in the usual way

.. nbplot::

    >>> solver = rr.FISTA(loss)
    >>> obj_vals = solver.fit()
    >>> solution = solver.composite.coefs

    /home/jb/code/regreg/regreg/smooth/glm.py:883: RuntimeWarning: overflow encountered in exp
      exp_x = np.exp(x)

Regularized logistic regression tutorial
========================================

The :math:`\ell_2` regularized logistic regression problem minimizes the
objective

.. math::


   -2\left(Y^TX\beta - \sum_i \log \left[ 1 + \exp(x_i^T\beta) \right] \right) + \lambda \|\beta\|_2^2

 which corresponds to the usual logistic regression model

.. math::


   P(Y_i=1) = \mbox{logit}(x_i^T\beta) = \frac{1}{1 + \mbox{exp}(-x_i^T\beta)}

To solve this problem using RegReg we begin by loading the necessary
numerical libraries

.. nbplot::

    >>> import numpy as np
    >>> import regreg.api as rr

The only code needed to add logistic regression is a class with one
method which computes the objective and its gradient.

.. literalinclude:: ../code/regreg/smooth.py .. :pyobject:
logistic\_deviance

Next, let's generate some example data,

.. nbplot::

    >>> X = np.random.normal(0,1,500000).reshape((500,1000))
    >>> Y = np.random.randint(0,2,500)

Now we can create the problem object, beginning with the loss function

.. nbplot::

    >>> loss = rr.glm.logistic(X,successes=Y)
    >>> penalty = rr.identity_quadratic(1., 0., 0., 0.)
    >>> loss.quadratic = penalty
    >>> loss

    /home/jb/code/regreg/regreg/smooth/glm.py:616: RuntimeWarning: divide by zero encountered in log
      loss_terms = np.log(saturated) * self.successes + np.log(1-saturated) * (self.trials - self.successes)
    /home/jb/code/regreg/regreg/smooth/glm.py:616: RuntimeWarning: invalid value encountered in multiply
      loss_terms = np.log(saturated) * self.successes + np.log(1-saturated) * (self.trials - self.successes)


.. math::

    \ell^{\text{logit}}\left(X_{}\beta\right)


The logistic log-likelihood function is written without a matrix
:math:``X``. We use the ".linear" to specify the linear composition
:math:``X\beta``. Similarly, we could use ".affine" to specify an offset
:math:``X\beta + \alpha``. The penalty contains the regularization
parameter that can be easily accessed and changed,

.. nbplot::

    >>> penalty.coef
    1.0

Next, we can select our algorithm of choice and use it solve the
problem,

.. nbplot::

    >>> solver = rr.FISTA(loss)
    >>> obj_vals = solver.fit(max_its=100, tol=1e-5)
    >>> solution = solver.composite.coefs

Here max\_its represents primal iterations, and tol is the primal
tolerance.

.. nbplot::

    >>> obj_vals
    array([ 346.57359028,  136.53950748,   90.2193755 ,   64.43466269,
             47.87470994,   37.00482319,   29.71842221,   24.76148798,
             21.34530689,   18.96643609,   17.29779533,   16.12281623,
             15.29536449,   14.71503362,   14.31167115,   14.03551815,
             13.85081314,   13.73156923,   13.65873774,   13.61827105,
             13.59977824,   13.59557754,   13.59557754,   13.59471918,
             13.59379186,   13.59253739,   13.59093917,   13.58898429,
             13.58666655,   13.58398891,   13.58096563,   13.57762413,
             13.57400608,   13.5701678 ,   13.56617948,   13.56212315,
             13.55808937,   13.55417258,   13.5504657 ,   13.54705425,
             13.54401051,   13.54138846,   13.53921964,   13.53751045,
             13.53624117,   13.53536727,   13.53482378])

Multinomial regression
======================

The multinomial regression problem minimizes the objective

.. math::


   -2\left[ \sum_{j=1}^{J-1} \sum_{k=1}^p \beta_{jk}\sum_{i=1}^n x_{ik}y_{ij}
    - \sum_{i=1}^n \log \left(1 + \mbox{exp}(x_i^T\beta_j) \right)\right]

 which corresponds to a baseline category logit model for :math:`J`
nominal categories (e.g. Agresti, p.g. 272). For :math:`i \ne J` the
probabilities are measured relative to a baseline category :math:`J`

.. math::


   \frac{P(\mbox{Category } i)}{P(\mbox{Category } J)} = \mbox{logit}(x^T\beta_i) = \frac{1}{1 + \mbox{exp}(-x^T\beta_i)}

To solve this problem using RegReg we begin by loading the necessary
numerical libraries

.. nbplot::

    >>> import numpy as np
    >>> import regreg.api as rr

The only code needed to add multinomial regression to RegReg is a class
with one method which computes the objective and its gradient.

Next, let's generate some example data. The multinomial counts will be
stored in a :math:`n \times J` array

.. nbplot::

    >>> J = 5
    >>> n = 1000
    >>> p = 50
    >>> X = np.random.standard_normal((n,p))
    >>> Y = np.random.randint(0,10,n*J).reshape((n,J))

Now we can create the problem object, beginning with the loss function.
The coefficients will be stored in a :math:`p \times J-1` array, and we
need to let RegReg know that the coefficients will be a 2d array instead
of a vector. We can do this by defining the input\_shape in a
linear\_transform object that multiplies by X,

.. nbplot::

    >>> multX = rr.linear_transform(X, input_shape=(p,J-1))
    >>> loss = rr.multinomial_loglike.linear(multX, counts=Y)

    /home/jb/code/regreg/regreg/smooth/glm.py:1110: RuntimeWarning: divide by zero encountered in log
      loss_terms = np.log(saturated) * self.counts
    /home/jb/code/regreg/regreg/smooth/glm.py:1110: RuntimeWarning: invalid value encountered in multiply
      loss_terms = np.log(saturated) * self.counts

Next, we can solve the problem

.. nbplot::

    >>> solver = rr.FISTA(loss)
    >>> solver.fit()
    array([ 36210.74359185,  36064.23731992,  36049.98463927,  36043.55378638,
            36039.3663041 ,  36036.65582838,  36035.08641646])

    /home/jb/code/regreg/regreg/smooth/glm.py:1126: RuntimeWarning: overflow encountered in exp
      exp_x = np.exp(x)


When :math:`J=2` this model should reduce to logistic regression. We can
easily check that this is the case by first fitting the multinomial
model

.. nbplot::

    >>> J = 2
    >>> Y = np.random.randint(0,10,n*J).reshape((n,J))
    >>> multX = rr.linear_transform(X, input_shape=(p,J-1))	
    >>> loss = rr.multinomial_loglike.linear(multX, counts=Y)
    >>> solver = rr.FISTA(loss)
    >>> solver.fit(tol=1e-6)
    >>> multinomial_coefs = solver.composite.coefs.flatten()

    /home/jb/code/regreg/regreg/smooth/glm.py:1109: RuntimeWarning: invalid value encountered in true_divide
      saturated = self.counts / (1. * self.trials[:,np.newaxis])
    /home/jb/code/regreg/regreg/smooth/glm.py:1110: RuntimeWarning: divide by zero encountered in log
      loss_terms = np.log(saturated) * self.counts
    /home/jb/code/regreg/regreg/smooth/glm.py:1110: RuntimeWarning: invalid value encountered in multiply
      loss_terms = np.log(saturated) * self.counts
    /home/jb/code/regreg/regreg/smooth/glm.py:1126: RuntimeWarning: overflow encountered in exp
      exp_x = np.exp(x)

and then the equivalent logistic regresison model

.. nbplot::

    >>> successes = Y[:,0]
    >>> trials = np.sum(Y, axis=1)
    >>> loss = rr.glm.logistic(X, successes=successes, trials=trials)
    >>> solver = rr.FISTA(loss)
    >>> solver.fit(tol=1e-6)
    >>> logistic_coefs = solver.composite.coefs

    /home/jb/code/regreg/regreg/smooth/glm.py:615: RuntimeWarning: invalid value encountered in true_divide
      saturated = self.successes / self.trials
    /home/jb/code/regreg/regreg/smooth/glm.py:616: RuntimeWarning: divide by zero encountered in log
      loss_terms = np.log(saturated) * self.successes + np.log(1-saturated) * (self.trials - self.successes)
    /home/jb/code/regreg/regreg/smooth/glm.py:616: RuntimeWarning: invalid value encountered in multiply
      loss_terms = np.log(saturated) * self.successes + np.log(1-saturated) * (self.trials - self.successes)

Finally we can check that the two models gave the same coefficients

.. nbplot::

    >>> print(np.linalg.norm(multinomial_coefs - logistic_coefs) / np.linalg.norm(logistic_coefs))

    3.55561768285e-16

.. nbplot::

    >>> import numpy as np
    >>> import regreg.api as rr
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> digits = datasets.load_digits()

    Populating the interactive namespace from numpy and matplotlib

Hinge loss
----------

The SVM can be parametrized various ways, one way to write it as a
regression problem is to use the hinge loss:

.. math::


   \ell(r) = \max(1-x, 0)

.. nbplot::

    >>> hinge = lambda x: np.maximum(1-x, 0)
    >>> fig = plt.figure(figsize=(9,6))
    >>> ax = fig.gca()
    >>> r = np.linspace(-1,2,100)
    >>> ax.plot(r, hinge(r))
    [...]



The SVM loss is then

.. math::


   \ell(\beta) = C \sum_{i=1}^n h(Y_i X_i^T\beta) + \frac{1}{2} \|\beta\|^2_2
   )

 where :math:`Y_i \in \{-1,1\}` and :math:`X_i \in \mathbb{R}^p` is one
of the feature vectors.

In regreg, the hinge loss can be represented by composition of some of
the basic atoms. Specifcally, let
:math:`g:\mathbb{R}^n \rightarrow \mathbb{R}` be the sum of positive
part function

.. math::


   g(z) = \sum_{i=1}^n\max(z_i, 0).

 Then,

.. math::


   \ell(\beta) = g\left(Y \cdot X\beta \right)

 where the product in the parentheses is elementwise multiplication.

.. nbplot::

    >>> linear_part = np.array([[-1.]])
    >>> offset = np.array([1.])
    >>> hinge_rep = rr.positive_part.affine(linear_part, offset, lagrange=1.)
    >>> hinge_rep



.. math::

    \lambda_{} \left(\sum_{i=1}^{p} (X_{}\beta - \alpha_{})_i^+\right)


Let's plot the loss to be sure it agrees with our original hinge.

.. nbplot::

    >>> ax.plot(r, [hinge_rep.nonsmooth_objective(v) for v in r])
    >>> fig




Here is a vectorized version.

.. nbplot::

    >>> N = 1000
    >>> P = 200
    >>>
    >>> Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
    >>> X = np.random.standard_normal((N,P))
    >>> #X[Y==1] += np.array([30,-20] + (P-2)*[0])[np.newaxis,:]
    >>> X -= X.mean(0)[np.newaxis, :]
    >>> hinge_vec = rr.positive_part.affine(-Y[:, None] * X, np.ones_like(Y), lagrange=1.)

.. nbplot::

    >>> beta = np.ones(X.shape[1])
    >>> hinge_vec.nonsmooth_objective(beta), np.maximum(1 - Y * X.dot(beta), 0).sum()
    (6097.8895639290613, 6097.8895639290613)

Smoothed hinge
--------------

For optimization, the hinge loss is not differentiable so it is often
smoothed first.

The smoothing is applicable to general functions of the form

.. math::


   g(X\beta-\alpha) = g_{\alpha}(X\beta)

 where :math:`g_{\alpha}(z) = g(z-\alpha)` and is determined by a small
quadratic term

.. math::


   q(z) = \frac{C_0}{2} \|z-x_0\|^2_2 + v_0^Tz + c_0.

.. nbplot::

    >>> epsilon = 0.5
    >>> smoothing_quadratic = rr.identity_quadratic(epsilon, 0, 0, 0)
    >>> smoothing_quadratic



.. math::

    \begin{equation*} \frac{L_{}}{2}\|\beta\|^2_2 \end{equation*} 


The quadratic terms are determined by four parameters with
:math:`(C_0, x_0, v_0, c_0)`.

Smoothing of the function by the quadratic :math:`q` is performed by
Moreau smoothing:

.. math::


   S(g_{\alpha},q)(\beta) = \sup_{z \in \mathbb{R}^p} z^T\beta - g^*_{\alpha}(z) - q(z)

 where

.. math::


   g^*_{\alpha}(z) = \sup_{\beta \in \mathbb{R}^p} z^T\beta - g_{\alpha}(\beta)

 is the convex (Fenchel) conjugate of the composition :math:`g` with the
translation by :math:`-\alpha`.

The basic atoms in ``regreg`` know what their conjugate is. Our hinge
loss, ``hinge_rep``, is the composition of an ``atom``, and an affine
transform. This affine transform is split into two pieces, the linear
part, stored as ``linear_transform`` and its offset stored as
``atom.offset``. It is stored with ``atom`` as ``atom`` needs knowledge
of this when computing proximal maps.

.. nbplot::

    >>> hinge_rep.atom



.. math::

    \lambda_{} \left(\sum_{i=1}^{p} (\beta - \alpha_{})_i^+\right)


.. nbplot::

    >>> hinge_rep.atom.offset
    array([-1.])

.. nbplot::

    >>> hinge_rep.linear_transform.linear_operator
    array([[-1.]])

As we said before, ``hinge_rep.atom`` knows what its conjugate is

.. nbplot::

    >>> hinge_conj = hinge_rep.atom.conjugate
    >>> hinge_conj



.. math::

    I^{\infty}(\left\|\beta\right\|_{\infty} + I^{\infty}\left(\min(\beta) \in [0,+\infty)\right)  \leq \delta_{}) + \left \langle \eta_{}, \beta \right \rangle


The notation :math:`I^{\infty}` denotes a constraint. The expression can
therefore be parsed as a linear function :math:`\eta^T\beta` plus the
function

.. math::


   g^*(z) = \begin{cases}
   0 & 0 \leq z_i \leq \delta \, \forall i \\
   \infty & \text{otherwise.}
   \end{cases}

The term :math:`\eta` is derived from ``hinge_rep.atom.offset`` and is
stored in ``hinge_conj.quadratic``.

.. nbplot::

    >>> hinge_conj.quadratic.linear_term
    array([-1.])

Now, let's look at the smoothed hinge loss.

.. nbplot::

    >>> smoothed_hinge_loss = hinge_rep.smoothed(smoothing_quadratic)
    >>> smoothed_hinge_loss



.. math::

     \sup_{u \in \mathbb{R}^{p} } \left[ \langle X_{}\beta, u \rangle - \left(I^{\infty}(\left\|u\right\|_{\infty} + I^{\infty}\left(\min(u) \in [0,+\infty)\right)  \leq \delta_{}) + \frac{L_{}}{2}\|u\|^2_2 + \left \langle \eta_{}, u \right \rangle \right) \right]


It is now a smooth function and its objective value and gradient can be
computed with ``smooth_objective``.

.. nbplot::

    >>> ax.plot(r, [smoothed_hinge_loss.smooth_objective(v, 'func') for v in r])
    >>> fig




.. nbplot::

    >>> less_smooth = hinge_rep.smoothed(rr.identity_quadratic(5.e-2, 0, 0, 0))
    >>> ax.plot(r, [less_smooth.smooth_objective(v, 'func') for v in r])
    >>> fig




Fitting the SVM
---------------

We can now minimize this objective.

.. nbplot::

    >>> smoothed_vec = hinge_vec.smoothed(rr.identity_quadratic(0.2, 0, 0, 0))
    >>> soln = smoothed_vec.solve(tol=1.e-12, min_its=100)

Sparse SVM
----------

We might want to fit a sparse version, adding a sparsifying penalty like
the LASSO. This yields the problem

.. math::


   \text{minimize}_{\beta} \ell(\beta) + \lambda \|\beta\|_1

.. nbplot::

    >>> penalty = rr.l1norm(smoothed_vec.shape, lagrange=20)
    >>> problem = rr.simple_problem(smoothed_vec, penalty)
    >>> problem



.. math::

    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &=  \sup_{u \in \mathbb{R}^{p} } \left[ \langle X_{1}\beta, u \rangle - \left(I^{\infty}(\left\|u\right\|_{\infty} + I^{\infty}\left(\min(u) \in [0,+\infty)\right)  \leq \delta_{1}) + \frac{L_{1}}{2}\|u\|^2_2 + \left \langle \eta_{1}, u \right \rangle \right) \right] \\
    g(\beta) &= \lambda_{2} \|\beta\|_1 \\
    \end{aligned}



.. nbplot::

    >>> sparse_soln = problem.solve(tol=1.e-12)
    >>> sparse_soln
    array([ 0.03304914, -0.        , -0.        ,  0.        ,  0.        ,
            0.        ,  0.03483583,  0.15968781, -0.        ,  0.06871071,
            0.        ,  0.04503846,  0.05386418, -0.        , -0.        ,
           -0.17794024, -0.00517231,  0.        ,  0.01035313,  0.        ,
           -0.02139741, -0.        ,  0.        ,  0.        , -0.04007206,
           -0.        , -0.        , -0.        , -0.18273235, -0.04084604,
           -0.        , -0.04247477,  0.        , -0.        , -0.        ,
           -0.        ,  0.        , -0.05138778, -0.        ,  0.05551519,
            0.        , -0.        ,  0.        ,  0.00054537, -0.        ,
            0.        ,  0.        ,  0.01061234, -0.        , -0.11423837,
           -0.00992783,  0.0379986 ,  0.01892455, -0.04748856,  0.02759452,
           -0.02679497,  0.        , -0.0157673 ,  0.0730639 , -0.03407619,
           -0.        , -0.        ,  0.        ,  0.        ,  0.        ,
            0.05225094,  0.0252666 ,  0.        ,  0.03610138, -0.05686105,
            0.17426776,  0.        , -0.04021431,  0.        ,  0.        ,
            0.        , -0.00416503, -0.        ,  0.00358612, -0.        ,
            0.        , -0.        ,  0.        ,  0.        , -0.        ,
            0.0467979 ,  0.16845895, -0.        ,  0.05867648,  0.12650904,
           -0.01900728,  0.        , -0.        , -0.        ,  0.05843162,
            0.02891906,  0.        ,  0.        ,  0.        , -0.00390111,
           -0.0399792 ,  0.0267627 , -0.11774134,  0.09287444,  0.01087837,
           -0.        ,  0.        ,  0.0693639 ,  0.13748441,  0.        ,
            0.        ,  0.14458581, -0.        ,  0.02308386, -0.03325608,
           -0.02064834, -0.06745733,  0.        ,  0.03952713,  0.        ,
           -0.11735477,  0.08387239,  0.        , -0.00501051, -0.        ,
           -0.        ,  0.        , -0.09095568, -0.01797337,  0.        ,
           -0.04250495,  0.        , -0.01043823, -0.        , -0.10192709,
           -0.01609947, -0.        ,  0.        ,  0.        , -0.02167176,
           -0.        ,  0.00178806, -0.01555563, -0.12500119, -0.        ,
            0.05612546,  0.        ,  0.        , -0.04199501, -0.        ,
           -0.11268056,  0.02722896,  0.04332363, -0.04927434,  0.        ,
           -0.03422833,  0.02818508, -0.        , -0.08347189,  0.02131943,
            0.        , -0.02748869,  0.03896059, -0.07176999,  0.        ,
            0.01112536,  0.03281508,  0.00293472, -0.        , -0.        ,
            0.        , -0.09897701,  0.        , -0.01186053, -0.        ,
            0.01332707,  0.        , -0.        ,  0.075996  , -0.20613866,
           -0.08273067, -0.        , -0.        ,  0.        , -0.        ,
           -0.        ,  0.        , -0.        ,  0.07824073, -0.05743607,
            0.0284728 ,  0.        , -0.01822421,  0.07680819,  0.02362438,
           -0.        ,  0.03540912, -0.08629176,  0.01724228, -0.        ])

What value of :math:`\lambda` should we use? For the :math:`\ell_1`
penalty in Lagrange form, the smallest :math:`\lambda` such that the
solution is zero can be found by taking the dual norm, the
:math:`\ell_{\infty}` norm, of the gradient of the smooth part at 0.

.. nbplot::

    >>> linf_norm = penalty.conjugate
    >>> linf_norm



.. math::

    I^{\infty}(\|\beta\|_{\infty} \leq \delta_{})


Just computing the conjugate will yield an :math:`\ell_{\infty}`
constraint, but this object can still be used to compute the desired
value of :math:`\lambda`.

.. nbplot::

    >>> score_at_zero = smoothed_vec.smooth_objective(np.zeros(smoothed_vec.shape), 'grad')
    >>> lam_max = linf_norm.seminorm(score_at_zero, lagrange=1.)
    >>> lam_max
    89.741253727018631

.. nbplot::

    >>> penalty.lagrange = lam_max * 1.001
    >>> problem.solve(tol=1.e-12, min_its=200)
    array([ 0., -0., -0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
           -0., -0., -0., -0.,  0.,  0.,  0., -0., -0.,  0.,  0., -0., -0.,
           -0.,  0., -0., -0., -0., -0.,  0., -0.,  0.,  0.,  0., -0., -0.,
            0.,  0., -0.,  0.,  0., -0.,  0., -0.,  0., -0., -0., -0.,  0.,
            0., -0.,  0., -0., -0., -0.,  0., -0., -0., -0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0., -0.,  0.,  0., -0.,  0.,  0.,  0., -0.,  0.,
            0., -0.,  0.,  0., -0.,  0., -0.,  0.,  0., -0.,  0.,  0., -0.,
            0., -0., -0.,  0.,  0.,  0., -0.,  0., -0., -0.,  0., -0.,  0.,
            0., -0., -0.,  0.,  0.,  0.,  0.,  0., -0.,  0., -0., -0., -0.,
            0.,  0.,  0., -0.,  0.,  0., -0., -0.,  0.,  0., -0., -0., -0.,
           -0.,  0., -0., -0., -0., -0., -0.,  0.,  0., -0., -0.,  0., -0.,
           -0., -0.,  0., -0.,  0., -0., -0., -0.,  0.,  0., -0.,  0., -0.,
            0., -0., -0.,  0.,  0., -0.,  0., -0.,  0.,  0.,  0.,  0., -0.,
           -0.,  0., -0.,  0., -0., -0.,  0., -0., -0.,  0., -0., -0., -0.,
           -0.,  0., -0., -0.,  0., -0.,  0., -0.,  0.,  0., -0.,  0.,  0.,
            0.,  0., -0.,  0., -0.])

.. nbplot::

    >>> penalty.lagrange = lam_max * 0.99
    >>> problem.solve(tol=1.e-12, min_its=200)
    array([ 0.        , -0.        , -0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        , -0.        , -0.        ,
           -0.22022518, -0.        ,  0.        ,  0.        ,  0.        ,
           -0.        , -0.        ,  0.        ,  0.        , -0.        ,
           -0.        , -0.        ,  0.        , -0.        , -0.        ,
           -0.        , -0.        ,  0.        , -0.        ,  0.        ,
            0.        ,  0.        , -0.        , -0.        ,  0.        ,
            0.        , -0.        ,  0.        ,  0.        , -0.        ,
            0.        , -0.        ,  0.        ,  0.        , -0.        ,
           -0.        ,  0.        ,  0.        , -0.        ,  0.        ,
           -0.        , -0.        , -0.        ,  0.        , -0.        ,
           -0.        , -0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        , -0.        ,
            0.        ,  0.        , -0.        ,  0.        ,  0.        ,
            0.        , -0.        ,  0.        ,  0.        , -0.        ,
            0.        ,  0.        , -0.        ,  0.        , -0.        ,
            0.        ,  0.        , -0.        ,  0.        ,  0.        ,
           -0.        ,  0.        , -0.        , -0.        ,  0.        ,
            0.        ,  0.        , -0.        ,  0.        , -0.        ,
           -0.        ,  0.        , -0.        ,  0.        ,  0.        ,
           -0.        , -0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        , -0.        ,  0.        , -0.        ,
           -0.        , -0.        ,  0.        ,  0.        ,  0.        ,
           -0.        ,  0.        ,  0.        , -0.        , -0.        ,
            0.        ,  0.        , -0.        , -0.        , -0.        ,
           -0.        ,  0.        , -0.        , -0.        , -0.        ,
           -0.        , -0.        ,  0.        ,  0.        , -0.        ,
           -0.        ,  0.        , -0.        , -0.        , -0.        ,
            0.        , -0.        ,  0.        , -0.        , -0.        ,
           -0.        ,  0.        ,  0.        , -0.        ,  0.        ,
           -0.        ,  0.        , -0.        , -0.        ,  0.        ,
            0.        , -0.        ,  0.        , -0.        ,  0.        ,
            0.        ,  0.        ,  0.        , -0.        , -0.        ,
            0.        , -0.        ,  0.        , -0.        , -0.        ,
            0.        , -0.        , -0.        ,  0.        , -0.18042994,
           -0.        , -0.        , -0.        ,  0.        , -0.        ,
           -0.        ,  0.        , -0.        ,  0.        , -0.        ,
            0.        ,  0.        , -0.        ,  0.        ,  0.        ,
            0.        ,  0.        , -0.        ,  0.        , -0.        ])

Path of solutions
~~~~~~~~~~~~~~~~~

If we want a path of solutions, we can simply take multiples of
``lam_max``. This is similar to the strategy that packages like
``glmnet`` use

.. nbplot::

    >>> path = []
    >>> lam_vals = (np.linspace(0.05, 1.01, 50) * lam_max)[::-1]
    >>> for lam_val in lam_vals:
    ...     penalty.lagrange = lam_val
    ...     path.append(problem.solve(min_its=200).copy())
    >>> fig = plt.figure(figsize=(12,8))
    >>> ax = fig.gca()
    >>> path = np.array(path)
    >>> ax.plot(path);




Changing the penalty
--------------------

We may not want to penalize features the same. We may want some features
to be unpenalized. This can be achieved by introducing possibly non-zero
feature weights to the :math:`\ell_1` norm

.. math::


   \beta \mapsto \sum_{j=1}^p w_j|\beta_j|

.. nbplot::

    >>> weights = np.random.sample(P) + 1.
    >>> weights[:5] = 0.
    >>> weighted_penalty = rr.weighted_l1norm(weights, lagrange=1.)
    >>> weighted_penalty



.. math::

    \lambda_{} \|W\beta\|_1


.. nbplot::

    >>> weighted_dual = weighted_penalty.conjugate
    >>> weighted_dual



.. math::

    I^{\infty}(\|W\beta\|_{\infty} \leq \delta_{})


.. nbplot::

    >>> lam_max_weight = weighted_dual.seminorm(score_at_zero, lagrange=1.)
    >>> lam_max_weight

    79.414068936028059

.. nbplot::

    >>> weighted_problem = rr.simple_problem(smoothed_vec, weighted_penalty)
    >>> path = []
    >>> lam_vals = (np.linspace(0.05, 1.01, 50) * lam_max_weight)[::-1]
    >>> for lam_val in lam_vals:
    ...     weighted_penalty.lagrange = lam_val
    ...     path.append(weighted_problem.solve(min_its=200).copy())
    >>> fig = plt.figure(figsize=(12,8))
    >>> ax = fig.gca()
    >>> path = np.array(path)
    >>> ax.plot(path);




Note that there are 5 coefficients that are not penalized hence they are
nonzero the entire path.

Group LASSO
^^^^^^^^^^^

Variables may come in groups. A common penalty for this setting is the
group LASSO. Let

.. math::


   \{1, \dots, p\} = \cup_{g \in G} g

 be a partition of the set of features and :math:`w_g` a weight for each
group. The group LASSO penalty is

.. math::


   \beta \mapsto \sum_{g \in G} w_g \|\beta_g\|_2.

.. nbplot::

    >>> groups = []
    >>> for i in range(P//5):
    ...     groups.extend([i]*5)
    >>> weights = dict([g, np.random.sample()+1] for g in np.unique(groups))
    >>> group_penalty = rr.group_lasso(groups, weights=weights, lagrange=1.)


.. nbplot::

    >>> group_dual = group_penalty.conjugate
    >>> lam_max_group = group_dual.seminorm(score_at_zero, lagrange=1.)

.. nbplot::

    >>> group_problem = rr.simple_problem(smoothed_vec, group_penalty)
    >>> path = []
    >>> lam_vals = (np.linspace(0.05, 1.01, 50) * lam_max_group)[::-1]
    >>> for lam_val in lam_vals:
    ...     group_penalty.lagrange = lam_val
    ...     path.append(group_problem.solve(min_its=200).copy())
    >>> fig = plt.figure(figsize=(12,8))
    >>> ax = fig.gca()
    >>> path = np.array(path)
    >>> ax.plot(path);




As expected, variables enter in groups here.

Bound form
~~~~~~~~~~

The common norm atoms also have a bound form. That is, we can just as
easily solve the problem

.. math::


   \text{minimize}_{\beta: \|\beta\|_1 \leq \delta}\ell(\beta)

.. nbplot::

    >>> bound_l1 = rr.l1norm(P, bound=2.)
    >>> bound_l1



.. math::

    I^{\infty}(\|\beta\|_1 \leq \delta_{})


.. nbplot::

    >>> bound_problem = rr.simple_problem(smoothed_vec, bound_l1)
    >>> bound_problem



.. math::

    
    \begin{aligned}
    \text{minimize}_{\beta} & f(\beta) + g(\beta) \\
    f(\beta) &=  \sup_{u \in \mathbb{R}^{p} } \left[ \langle X_{1}\beta, u \rangle - \left(I^{\infty}(\left\|u\right\|_{\infty} + I^{\infty}\left(\min(u) \in [0,+\infty)\right)  \leq \delta_{1}) + \frac{L_{1}}{2}\|u\|^2_2 + \left \langle \eta_{1}, u \right \rangle \right) \right] \\
    g(\beta) &= I^{\infty}(\|\beta\|_1 \leq \delta_{2}) \\
    \end{aligned}



.. nbplot::

    >>> bound_soln = bound_problem.solve()
    >>> np.fabs(bound_soln).sum()
    2.0000000000000004

