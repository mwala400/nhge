"""
NHGE Command Line Interface

Usage:
    python -m nhge --version
    python -m nhge --info
"""

import argparse
from nhge import __version__, __author__, __license__


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Harmonic Graph Engine (NHGE)",
        prog="nhge"
    )
    parser.add_argument("--version", action="version", version=f"NHGE {__version__}")
    parser.add_argument("--info", action="store_true", help="Show package information")

    args = parser.parse_args()

    if args.info:
        print(f"NHGE version     : {__version__}")
        print(f"Author           : {__author__}")
        print(f"License          : {__license__}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()