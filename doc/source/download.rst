.. _download:

Downloading and installing RegReg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RegReg source code is hosted at

http://github.com/regreg/regreg

RegReg depends on the following Python tools:

* numpy_
* scipy_
* Cython_

You can clone the RegReg github repo using::

     git clone https://github.com/regreg/regreg.git

RegReg is built with meson-python_. Installation is a call to pip::

     cd regreg
     pip install .

For development, an editable install rebuilds the Cython extensions
on import. The build tools must be installed in the environment first::

     pip install meson-python meson ninja cython numpy setuptools_scm
     pip install --no-build-isolation -e .

or, with uv_, simply::

     uv sync

That's it!

.. _meson-python: https://meson-python.readthedocs.io
.. _uv: https://docs.astral.sh/uv/

Testing your installation
-------------------------

There is a small but growing suite of tests that can be run with pytest_::

     pytest regreg

.. _pytest: https://docs.pytest.org

.. include:: links_names.inc
