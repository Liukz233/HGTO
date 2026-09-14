"""Command-line entry points for HGTO optimization and evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def load_settings(path, *, method=None, device=None, state_device=None, max_updates=None):
    """Resolve a configuration without constructing a mesh or creating output."""
    settings = yaml.safe_load(Path(path).read_text())
    if not isinstance(settings, dict) or not isinstance(settings.get("case"), dict):
        raise ValueError("A configuration must define a case mapping")
    selected = method or settings.get("method", "hgto")
    if selected not in ("hgto", "oc"):
        raise ValueError("Use hgto or oc; NTopo has a separate TensorFlow launcher")
    if selected not in settings:
        raise ValueError(f"This configuration does not define a {selected} comparison")
    category = settings.get("category", "linear2d")
    if category not in ("linear2d", "linear3d", "domains", "nonlinear"):
        raise ValueError(f"Unknown category: {category}")
    settings["method"] = selected
    options = settings[selected]
    if not isinstance(options, dict):
        raise ValueError(f"{selected} settings must be a mapping")
    if device:
        if selected == "oc" and device != "cpu":
            raise ValueError("SIMP--OC uses CPU mechanics; choose --device cpu")
        if selected != "oc" or category != "nonlinear":
            options["device"] = device
    if state_device:
        if selected != "hgto" or category != "nonlinear":
            raise ValueError("--state-device applies to nonlinear HGTO only")
        options["state_device"] = state_device
    if max_updates is not None:
        if max_updates < 1:
            raise ValueError("--max-updates must be positive")
        if selected == "oc" and category != "nonlinear":
            raise ValueError("Linear OC budgets are set by max_stage_steps in its configuration")
        options["max_updates"] = max_updates
    if category == "nonlinear":
        from hgto.nonlinear.runner import validate_settings

        validate_settings(settings)
    elif selected == "hgto":
        from hgto.optimization import GraphOptimizerConfig

        GraphOptimizerConfig(**options)
    else:
        from hgto.reference.oc import OCConfig

        OCConfig(**{k: v for k, v in options.items() if k != "device"})
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Optimize a configured problem")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--method", choices=("hgto", "oc"))
    run.add_argument("--device", help="Density network device; cpu or cuda:0")
    run.add_argument("--state-device", help="Optional nonlinear mechanics device")
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--max-updates", type=int)
    run.add_argument(
        "--dry-run", action="store_true", help="Validate and print settings without running"
    )
    plot = sub.add_parser("plot", help="Render a saved physical density")
    plot.add_argument("directory", type=Path)
    plot.add_argument("--output", required=True, type=Path)
    evaluate = sub.add_parser("evaluate", help="Independently evaluate a saved design on the CPU")
    evaluate.add_argument("directory", type=Path)
    compare = sub.add_parser(
        "compare-plastic",
        help="Evaluate elastic/plastic designs through common loading and unloading",
    )
    compare.add_argument("--elastic", required=True, type=Path)
    compare.add_argument("--plastic", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.threads < 1:
                raise ValueError("--threads must be positive")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[name] = str(args.threads)
            settings = load_settings(
                args.config,
                method=args.method,
                device=args.device,
                state_device=args.state_device,
                max_updates=args.max_updates,
            )
            if args.dry_run:
                print(yaml.safe_dump(settings, sort_keys=False))
                return
            if args.output.exists():
                raise FileExistsError(f"Choose a new output directory: {args.output}")
            import torch

            for key in ("device", "state_device"):
                device = settings[settings["method"]].get(key, "cpu")
                if str(device).startswith("cuda") and not torch.cuda.is_available():
                    raise ValueError(
                        "CUDA is unavailable; use --device cpu (and --state-device cpu if set)"
                    )
            if settings.get("category") == "nonlinear":
                from hgto.nonlinear.runner import run as optimize

                result = optimize(settings, args.output, threads=args.threads)
            else:
                from hgto.linear import run as optimize

                result = optimize(
                    settings, args.output, args.config.resolve().parent, threads=args.threads
                )["summary"]
            print(json.dumps(result, indent=2))
        elif args.command == "plot":
            from hgto.plotting import plot_design

            plot_design(args.directory, args.output)
            print(args.output)
        elif args.command == "evaluate":
            from hgto.evaluation import evaluate_design

            print(json.dumps(evaluate_design(args.directory), indent=2))
        else:
            from hgto.nonlinear.response import compare_designs

            if args.output.exists():
                raise FileExistsError(f"Choose a new output directory: {args.output}")
            print(json.dumps(compare_designs(args.elastic, args.plastic, args.output), indent=2))
    except (ValueError, FileExistsError, FileNotFoundError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
