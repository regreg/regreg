''' Distutils / setuptools helpers

Copied from nibabel 'nisext.sexts'
'''

from distutils.version import LooseVersion
from distutils import log

# Dependency checks
def package_check(pkg_name, version=None,
                  optional=False,
                  checker=LooseVersion,
                  version_getter=None,
                  messages=None
                  ):
    ''' Check if package `pkg_name` is present, and correct version

    Parameters
    ----------
    pkg_name : str
       name of package as imported into python
    version : {None, str}, optional
       minimum version of the package that we require. If None, we don't
       check the version.  Default is None
    optional : {False, True}, optional
       If False, raise error for absent package or wrong version;
       otherwise warn
    checker : callable, optional
       callable with which to return comparable thing from version
       string.  Default is ``distutils.version.LooseVersion``
    version_getter : {None, callable}:
       Callable that takes `pkg_name` as argument, and returns the
       package version string - as in::

          ``version = version_getter(pkg_name)``

       If None, equivalent to::

          mod = __import__(pkg_name); version = mod.__version__``
    messages : None or dict, optional
       dictionary giving output messages
    '''
    if version_getter is None:
        def version_getter(pkg_name):
            mod = __import__(pkg_name)
            return mod.__version__
    if messages is None:
        messages = {}
    msgs = {
         'missing': 'Cannot import package "%s" - is it installed?',
         'missing opt': 'Missing optional package "%s"',
         'opt suffix' : '; you may get run-time errors',
         'version too old': 'You have version %s of package "%s"'
                            ' but we need version >= %s', }
    msgs.update(messages)
    try:
        __import__(pkg_name)
    except ImportError:
        if not optional:
            raise RuntimeError(msgs['missing'] % pkg_name)
        log.warn(msgs['missing opt'] % pkg_name +
                 msgs['opt suffix'])
        return
    if not version:
        return
    try:
        have_version = version_getter(pkg_name)
    except AttributeError:
        raise RuntimeError('Cannot find version for %s' % pkg_name)
    if checker(have_version) < checker(version):
        if optional:
            log.warn(msgs['version too old'] % (have_version,
                                                pkg_name,
                                                version)
                     + msgs['opt suffix'])
        else:
            raise RuntimeError(msgs['version too old'] % (have_version,
                                                          pkg_name,
                                                          version))
