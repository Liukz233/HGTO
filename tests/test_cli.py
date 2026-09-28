"""Public commands exercise configuration, saved output, and physical reanalysis."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hgto.cli import load_settings

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run(
        [sys.executable, "-m", "hgto", *map(str, args)], cwd=ROOT, text=True, capture_output=True
    )


def test_all_paper_configurations_resolve():
    for config in (ROOT / "configs").rglob("*.yaml"):
        resolved = load_settings(config)
        assert resolved["method"] == "hgto"
        if "oc" in resolved:
            assert load_settings(config, method="oc")["method"] == "oc"


def test_cpu_command_saves_a_feasible_independently_evaluated_design(tmp_path):
    output = tmp_path / "design"
    config = ROOT / "configs/smoke/cantilever_cpu.yaml"
    result = command("run", "--config", config, "--output", output)
    assert result.returncode == 0, result.stderr
    record = json.loads((output / "result.json").read_text())
    assert record["optimizer"] == "Adam"
    assert record["independent_relative_C_error"] < 1e-7
    assert record["volume"] == pytest.approx(0.5, abs=1e-10)
    rho = np.load(output / "rho.npy")
    assert np.isfinite(rho).all()
    evaluation = command("evaluate", output)
    assert evaluation.returncode == 0, evaluation.stderr
    value = json.loads(evaluation.stdout)
    assert value["compliance"] == pytest.approx(record["C_raw"], rel=1e-8)
    assert command("plot", output, "--output", tmp_path / "design.png").returncode == 0
    before = (output / "rho.npy").read_bytes()
    repeated = command("run", "--config", config, "--output", output)
    assert repeated.returncode != 0
    assert (output / "rho.npy").read_bytes() == before


def test_dry_run_resolves_nonlinear_oc_cuda_and_rejects_conflicting_devices(tmp_path):
    output = tmp_path / "unused"
    config = ROOT / "configs/nonlinear/bridge.yaml"
    assert command("run", "--config", config, "--output", output, "--dry-run").returncode == 0
    resolved = command(
        "run",
        "--config",
        config,
        "--output",
        output,
        "--method",
        "oc",
        "--device",
        "cuda:0",
        "--dry-run",
    )
    assert resolved.returncode == 0, resolved.stderr
    settings = load_settings(config, method="oc", device="cuda:0")
    assert settings["oc"]["device"] == "cuda:0"
    assert settings["oc"]["tangent_backend"] == "cudss"
    assert load_settings(config, method="oc")["oc"]["tangent_backend"] == "pypardiso"
    cpu = load_settings(config, method="oc", device="cpu")["oc"]
    assert (cpu["device"], cpu["tangent_backend"]) == ("cpu", "pypardiso")
    invalid = command(
        "run",
        "--config",
        config,
        "--output",
        output,
        "--method",
        "oc",
        "--device",
        "cuda:0",
        "--state-device",
        "cpu",
        "--dry-run",
    )
    assert invalid.returncode != 0
    assert not output.exists()


def test_nonlinear_adaptive_oc_cli_roundtrip(tmp_path):
    import torch
    import yaml

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for nonlinear OC command integration")
    config = tmp_path / "adaptive.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "category": "nonlinear",
                "method": "oc",
                "case": {"name": "cantilever_nh", "physics": "nh", "load": 0.00125},
                "oc": {
                    "objective": "complementary_work",
                    "continuation_updates": 0,
                    "max_updates": 1,
                    "load_steps": 2,
                    "adaptive_load": True,
                    "mixed_sign_update": "reciprocal_linear",
                },
            }
        )
    )
    out = tmp_path / "nonlinear"
    result = command("run", "--config", config, "--output", out, "--device", "cuda:0")
    assert result.returncode == 0, result.stderr
    protocol = json.loads((out / "protocol.json").read_text())
    verified = json.loads((out / "verification/evaluation.json").read_text())
    final = json.loads((out / "result.json").read_text())
    assert protocol["state_device"] == "cuda:0"
    assert verified["adaptive_load"] is True and verified["load_steps"] == 2
    assert final["training_to_validation_relative_error"] < 1e-9
