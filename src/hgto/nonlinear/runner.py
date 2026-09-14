"""Run the paper's nonlinear HGTO or SIMP--OC configuration."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
import yaml

from .optimization import run as optimize_graph
from .oc import NonlinearOCConfig, run as optimize_oc


def validate_settings(settings):
    case = settings["case"]
    expected = {
        "cantilever_nh": ("nh",),
        "bridge_nh": ("nh",),
        "connection_j2": ("elastic_j2", "j2"),
    }
    if case.get("name") not in expected or case.get("physics") not in expected[case["name"]]:
        raise ValueError("The nonlinear case and material model do not match")
    if case.get("load", 0) <= 0:
        raise ValueError("The load must be positive")
    method = settings.get("method", "hgto")
    allowed = (
        set(inspect.signature(optimize_graph).parameters) - {"name", "physics", "load", "output"}
        if method == "hgto"
        else {f.name for f in fields(NonlinearOCConfig)}
    )
    unknown = set(settings[method]) - allowed
    if unknown:
        raise ValueError(f"Unknown {method} options: {sorted(unknown)}")


def run(settings, output, *, threads=2):
    validate_settings(settings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    method = settings.get("method", "hgto")
    case = settings["case"]
    options = dict(settings[method])
    device = options.get("device", "cpu")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if method == "hgto":
        optimize_graph(case["name"], case["physics"], case["load"], output, **options)
    else:
        optimize_oc(
            case["name"], case["physics"], case["load"], output, NonlinearOCConfig(**options)
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
