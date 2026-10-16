#!/usr/bin/env python3
"""
Print the package version from git metadata (or PKG-INFO in an sdist).

Called from meson.build. Meson runs Python scripts with its own interpreter,
which under build isolation is the one that has setuptools_scm installed.

Untagged commits get a development version such as ``0.2.1.dev3``, without
the ``+g<hash>`` local part, which PyPI and TestPyPI do not accept.
"""
from setuptools_scm import get_version

print(get_version(local_scheme='no-local-version'))
