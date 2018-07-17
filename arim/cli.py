"""
arim - array imaging toolbox


"""

import argparse
from .__init__ import __version__

import sys


def print_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def print_version():
    print("arim version {}".format(__version__))


def main():
    """
    Entry point for Command Line Interface.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="print software version", action="store_true")
    args = parser.parse_args()

    if args.version:
        print_version()
        sys.exit(0)
    else:
        parser.print_usage()
        print()
        print_err("error: no commands supplied")
        sys.exit(-1)


if __name__ == "__main__":
    main()
