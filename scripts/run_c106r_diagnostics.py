"""Execute one recorded R0/R1 stage. No long training or final test entry point."""
import argparse
from pathlib import Path
from neurokinematics.neural import c106r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "runtime", "audit-data", "overfit"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = {"preflight": "preflight.json", "runtime": "runtime.json", "audit-data": "data-audit.json", "overfit": "overfit.json"}[args.action]
    if (args.output / artifact).exists():
        raise FileExistsError("preserve prior attempt: " + str(args.output / artifact))
    result = getattr(c106r, args.action.replace("-", "_"))(args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
