""" This file contains defines parameters for regreg that we use to fill
settings in setup.py, the regreg top-level docstring, and for building the docs.
In setup.py in particular, we exec this file, so it cannot import regreg
"""

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description  = 'A multi-algorithm Python framework for regularized regression'

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = \
"""
======
RegReg
======

RegReg is a simple multi-algorithm Python framework for prototyping and solving
regularized regression problems such as the LASSO. The goal is to enable
practitioners to quickly and easily experiment with a variety of different
models and choices of regularization.  In that spirit, the emphasis is on the
flexibility of the framework instead of computational speed for any particular
problem, though the speed tradeoff will generally not be too bad.
"""

# Minimum package versions
# Check against requirements.txt and .travis.yml
NUMPY_MIN_VERSION='1.6.0'
SCIPY_MIN_VERSION = '0.9'
CYTHON_MIN_VERSION = '0.18'

NAME                = 'regreg'
MAINTAINER          = "regreg developers"
MAINTAINER_EMAIL    = ""
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://github.org/regreg/regreg"
DOWNLOAD_URL        = ""#"http://github.com/regreg/regreg/archives/master"
LICENSE             = "BSD license"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "regreg developers"
AUTHOR_EMAIL        = ""#"regreg-devel@neuroimaging.scipy.org"
PLATFORMS           = "OS Independent"
STATUS              = 'alpha'
PROVIDES            = ["regreg"]
REQUIRES            = ["numpy (>=%s)" % NUMPY_MIN_VERSION,
                       "scipy (>=%s)" % SCIPY_MIN_VERSION]
