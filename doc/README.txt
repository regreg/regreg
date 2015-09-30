====================
Regreg Documentation
====================

This is the top level build directory for the regreg documentation.  All
of the documentation is written using Sphinx_, a Python documentation
system built on top of reST_.  In order to build the documentation,
you must have Sphinx v1.0 or greater installed.

This directory contains:

* Makefile - the build script to build the HTML or PDF docs. Type
  ``make help`` for a list of options.

* source - the directory containing the reST source

* source/faq - frequently asked questions

* source/api - placeholders to automatically generate the api documentation

* source/links_names.inc - reST document with hyperlink targets for common
  links used throughout the documentation

* source/conf.py - the sphinx configuration.

* source/sphinxext - some extensions to sphinx to handle math numpy_ docstring
  parsing;

* source/_static - used by the sphinx build system.

* source/_templates - used by the sphinx build system.


Building the documentation
--------------------------

::

    make html

.. Since this README.txt is not processed by Sphinx during the
.. documentation build, I've included the links directly so it is at
.. least a valid reST doc.

.. _Sphinx: http://sphinx.pocoo.org/
.. _reST: http://docutils.sourceforge.net/rst.html
.. _numpy: http://www.scipy.org/NumPy
