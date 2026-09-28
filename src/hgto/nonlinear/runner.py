"""Run the paper's nonlinear HGTO or SIMP--OC configuration."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import torch
import yaml

from .optimization import validate_nh_solver, case as make_case, run as optimize_graph
from .oc import NonlinearOCConfig, run as optimize_oc
from hgto.stopping import PhysicalStopping, StopConfig


CASE_OPTIONS = {"name", "physics", "load", "refine", "nh_solver"}


def validate_settings(settings):
    case = settings["case"]
    unknown_case = set(case) - CASE_OPTIONS
    if unknown_case:
        raise ValueError(f"Unknown nonlinear case options: {sorted(unknown_case)}")
    expected = {
        "cantilever_nh": ("nh",),
        "bridge_nh": ("nh",),
        "lbracket_nh": ("nh",),
        "connection_j2": ("elastic_j2", "j2"),
    }
    if case.get("name") not in expected or case.get("physics") not in expected[case["name"]]:
        raise ValueError("The nonlinear case and material model do not match")
    if not math.isfinite(case.get("load", 0)) or case.get("load", 0) <= 0:
        raise ValueError("The load must be positive")
    refine = case.get("refine", 1)
    if type(refine) is not int or refine < 1:
        raise ValueError("refine must be a positive integer")
    if case.get("nh_solver") is not None:
        if case["physics"] != "nh":
            raise ValueError("nh_solver applies to Neo-Hookean cases only")
        validate_nh_solver(case["nh_solver"])
    method = settings.get("method", "hgto")
    if method not in ("hgto", "oc"):
        raise ValueError("Unknown nonlinear method")
    allowed = (
        set(inspect.signature(optimize_graph).parameters)
        - {"name", "physics", "load", "output", "setup", "spec"}
        if method == "hgto"
        else {f.name for f in fields(NonlinearOCConfig)}
    )
    unknown = set(settings[method]) - allowed
    if unknown:
        raise ValueError(f"Unknown {method} options: {sorted(unknown)}")
    if method == "hgto" and settings[method].get("stop_config") is not None:
        stopping = settings[method]["stop_config"]
        if isinstance(stopping, dict):
            stopping = StopConfig(**stopping)
        if not isinstance(stopping, StopConfig):
            raise ValueError("stop_config must be a stopping-configuration mapping")
        PhysicalStopping(stopping)
    if method == "oc":
        NonlinearOCConfig(**settings[method]).validate()


def run(settings, output, *, threads=2):
    validate_settings(settings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    method = settings.get("method", "hgto")
    case = settings["case"]
    options = dict(settings[method])
    if method == "hgto" and isinstance(options.get("stop_config"), dict):
        options["stop_config"] = StopConfig(**options["stop_config"])
    device = options.get("device", "cpu")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    case_options = {key: case[key] for key in ("refine", "nh_solver") if key in case}
    if method == "hgto" and options.get("nh_transition_beta") is not None:
        case_options["nh_transition_beta"] = options["nh_transition_beta"]
    setup, spec = make_case(case["name"], **case_options)
    if method == "hgto":
        optimize_graph(
            case["name"], case["physics"], case["load"], output, **options, setup=setup, spec=spec
        )
    else:
        optimize_oc(
            case["name"],
            case["physics"],
            case["load"],
            output,
            NonlinearOCConfig(**options),
            setup=setup,
            spec=spec,
        )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    record = json.loads((output / "record.json").read_text())
    record.update(wall_s_including_setup=elapsed, cpu_threads=threads)
    (output / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    package = Path(__file__).resolve().parents[1]
    manifest = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if case["physics"] == "nh":
        from .evaluation import finalize

        return finalize(output, "HGTO" if method == "hgto" else "SIMP--OC")
    # Plastic loading/unloading is evaluated jointly with compare-plastic.
    result = dict(
        record["final"],
        method=method,
        case=case["name"],
        termination=record["termination"],
        converged=record["converged"],
        design_updates=record["design_updates"],
        optimization_s=elapsed,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
