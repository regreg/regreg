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

     git clone git://github.com/regreg/regreg.git

Then installation is a simple call to python::

     cd regreg
     python setup.py install --user

That's it!

Testing your installation
-------------------------

There is a small but growing suite of tests that be easily checked using nose_::

     nosetests regreg

.. include:: links_names.inc
