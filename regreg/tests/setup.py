"""
RegReg: A package to solve regularized regression problems
"""

import os, sys
import string

from Cython.Compiler import Main

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('tests',parent_package,top_path)
    config.add_subpackage('atoms')
    return config

if __name__ == '__main__':


    
    from numpy.distutils.core import setup

    c = configuration(top_path='',
                      ).todict()
    setup(**c)
