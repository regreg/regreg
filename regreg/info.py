""" This file contains defines parameters for regreg that we use to fill
settings in setup.py, the regreg top-level docstring, and for building the docs.
In setup.py in particular, we exec this file, so it cannot import regreg
"""

# regreg version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 0
_version_minor = 0
_version_micro = 1
_version_extra = '.dev'

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

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

# versions
NUMPY_MIN_VERSION='1.3'
SCIPY_MIN_VERSION = '0.5'
CYTHON_MIN_VERSION = '0.11.1'

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
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
ISRELEASE           = _version_extra == ''
VERSION             = __version__
STATUS              = 'alpha'
PROVIDES            = ["regreg"]
REQUIRES            = ["numpy (>=%s)" % NUMPY_MIN_VERSION,
                       "scipy (>=%s)" % SCIPY_MIN_VERSION]
