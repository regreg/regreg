######
RegReg
######

RegReg is a simple multi-algorithm Python framework for prototyping and solving
regularized regression problems such as the LASSO. The goal is to enable
practitioners to quickly and easily experiment with a variety of different
models and choices of regularization.  In that spirit, the emphasis is on the
flexibility of the framework instead of computational speed for any particular
problem, though the speed trade-off will generally not be too bad.

************
Installation
************

RegReg needs Python 3.10 or later. Install the latest release from PyPI with::

    pip install regreg

There are wheels for Linux (x86_64), macOS (arm64) and Windows (64-bit); on
other platforms pip builds the Cython extensions from the source
distribution, which needs a C compiler.

The optional extras are ``regreg[sklearn]`` (scikit-learn, for the
``regreg.sklearn`` estimators and cross-validation splitters) and
``regreg[test]``. To install the development version from GitHub::

    pip install git+https://github.com/regreg/regreg

See ``INSTALL`` for building from a clone and running the tests.

***************
A first example
***************

The LASSO, a squared-error loss plus an l1 penalty:

.. code-block:: python

    import numpy as np
    import regreg.api as rr

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 20))
    Y = X[:, :3] @ [3, -2, 1.5] + rng.standard_normal(100)

    loss = rr.squared_error(X, Y)
    penalty = rr.l1norm(20, lagrange=20.)
    coef = rr.simple_problem(loss, penalty).solve(tol=1e-10)

*************
Documentation
*************

https://regreg.github.io/regreg/ (with the API reference at
https://regreg.github.io/regreg/api/)

****
Code
****

See https://github.com/regreg/regreg

Released under the BSD two-clause license - see the file ``LICENSE`` in the
source distribution.

The latest released version is at https://pypi.org/project/regreg/; the
changes in each release are listed in the file ``Changelog``.

*******
Support
*******

Please put up issues on the `regreg issue tracker
<https://github.com/regreg/regreg/issues>`_.
