DESCRIP = 'Convert markdown notebook templates into built markdown'
EPILOG = ''

import os
from os.path import (abspath, join as pjoin, basename, splitext, getmtime,
                     exists)
from glob import glob
from subprocess import check_call

from argparse import ArgumentParser, RawDescriptionHelpFormatter


def get_parser():
    parser = ArgumentParser(description=DESCRIP,
                            epilog=EPILOG,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('in_dir', type=str, help='input directory')
    parser.add_argument('nb_dir', type=str, help='notebook output directory')
    parser.add_argument('--if-newer', action='store_true',
                        help='Only write markdown output if input is newer '
                        'than markdown output')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    in_files = [abspath(f) for f in glob(pjoin(args.in_dir, '*.md'))]
    os.chdir(args.nb_dir)
    for in_file in in_files:
        froot = splitext(basename(in_file))[0]
        nb_file = froot + '.ipynb'
        md_file = froot + '.md'
        if args.if_newer and exists(md_file):
            if getmtime(md_file) > getmtime(in_file):
                continue
        check_call(['notedown', in_file, '-o', nb_file])
        check_call(['jupyter', 'nbconvert', '--execute', '--inplace',
                    nb_file, '--ExecutePreprocessor.timeout=-1'])
        check_call(['jupyter', 'nbconvert', '--to=markdown',
                    '--output=' + md_file, nb_file])


if __name__ == '__main__':
    main()
