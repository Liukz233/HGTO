"""Run a named group of paper configurations sequentially."""

import argparse
from pathlib import Path
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=["linear2d", "domains", "linear3d", "nonlinear", "all"], required=True
    )
    parser.add_argument("--method", choices=["hgto", "oc"], default="hgto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suites = (
        ["linear2d", "domains", "linear3d", "nonlinear"] if args.suite == "all" else [args.suite]
    )
    for suite in suites:
        for config in sorted((root / "configs" / suite).glob("*.yaml")):
            if args.method not in yaml.safe_load(config.read_text()):
                continue
            output = args.output / suite / config.stem / args.method
            command = [
                sys.executable,
                "-m",
                "hgto",
                "run",
                "--config",
                str(config),
                "--method",
                args.method,
                "--output",
                str(output),
                "--threads",
                "2",
            ]
            if args.device:
                command += ["--device", args.device]
            print(f"{suite}/{config.stem}: {args.method} -> {output}", flush=True)
            if args.dry_run:
                command.append("--dry-run")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
