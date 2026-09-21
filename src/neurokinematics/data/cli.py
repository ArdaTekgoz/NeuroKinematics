"""Command line interface for the frozen F0-04 data factory."""

import argparse
from pathlib import Path

from .factory import generate_dataset, verify_dataset


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--evidence", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = generate_dataset(args.output, args.evidence) if args.command == "generate" else verify_dataset(args.output, args.manifest)
    print(result)


if __name__ == "__main__":
    main()
