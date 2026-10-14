#!/usr/bin/env python3
"""
Print the package version from git metadata (or PKG-INFO in an sdist).

Called from meson.build. Meson runs Python scripts with its own interpreter,
which under build isolation is the one that has setuptools_scm installed.
"""
from setuptools_scm import get_version

print(get_version())
