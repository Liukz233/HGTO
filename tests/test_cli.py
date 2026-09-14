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


def test_dry_run_and_invalid_device_do_not_create_output(tmp_path):
    output = tmp_path / "unused"
    config = ROOT / "configs/nonlinear/bridge.yaml"
    assert command("run", "--config", config, "--output", output, "--dry-run").returncode == 0
    invalid = command(
        "run", "--config", config, "--output", output, "--method", "oc", "--device", "cuda:0"
    )
    assert invalid.returncode != 0
    assert not output.exists()
