""" Check R packages are installed
"""
from __future__ import print_function

import sys
import rpy2.robjects.packages as rpackages


def check_r_packages(packnames):
    """ Check R packages in list `packnames` are installed

    Return empty string if all packages are installed, otherwise, informative
    error message.
    """
    missing_packnames = []
    for packname in packnames:
        if not rpackages.isinstalled(packname):
            missing_packnames.append(packname)
    if len(missing_packnames) == 0:
        return ''
    raise RuntimeError("Please install R package(s) " +
                       ', '.join(missing_packnames))


def main():
    check_r_packages(sys.argv[1:])


if __name__ == '__main__':
    main()
