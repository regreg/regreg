""" Check R packages are installed
"""
from __future__ import print_function

import sys
import rpy2.robjects.packages as rpackages


def main():
    missing_packnames = []
    for packname in sys.argv[1:]:
        if not rpackages.isinstalled(packname):
            missing_packnames.append(packname)
    if len(missing_packnames):
        print("Please install R package(s)", ','.join(missing_packnames))
        sys.exit(1)


if __name__ == '__main__':
    main()
