"""Unified command-line entry point for EviSET.

Examples:
    python main.py preprocess --raw-dir raw_data --out-dir processed_data --overwrite
    python main.py preprocess --subject S01 S02 --task 1 3 --overwrite
    python main.py test --data-dir processed_data --out-dir results --protocol both
    python main.py test --subject S01 S02 --task 1 3 --model SVM EEGNet --protocol within
    python main.py test --data-dir processed_data --out-dir results_smoke --protocol within --quick --methods SVM
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preprocess", "test"], help="Task to run.")
    return parser.parse_args(sys.argv[1:2])


def main() -> None:
    args = parse_args()
    if args.command == "preprocess":
        from task.preprocess import main as preprocess_main

        sys.argv = ["preprocess"] + sys.argv[2:]
        preprocess_main()
    elif args.command == "test":
        from task.test import main as test_main

        sys.argv = ["test"] + sys.argv[2:]
        test_main()
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
