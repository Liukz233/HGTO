from hgto.reference.oc import OCConfig, optimize_oc
from hgto.linear2d.problems import make_problem
from hgto.reference import ScikitFEMElasticity
import pytest


@pytest.mark.parametrize("final_beta, expected_convergence", [(1.0, True), (8.0, False)])
def test_intermediate_stage_cap_does_not_claim_intermediate_convergence(
    final_beta, expected_convergence
):
    p = make_problem("cantilever", 8, 4, volume=0.4, radius=1.5)
    config = OCConfig(
        betas=(1.0, 2.0, final_beta),
        penalties=(1.0, 2.0, 3.0),
        max_stage_steps=300,
        min_stage_steps=1,
        stable_steps=1,
        change_tolerance=0.02,
        stopping_rule="density_change",
        continuation_stage_cap=2,
        solver="scipy",
    )
    out = optimize_oc(p, config)
    summary = out["summary"]
    assert [x["steps"] for x in summary["stages"][:2]] == [2, 2]
    assert not summary["stages"][0]["converged"]
    assert summary["converged"] is expected_convergence
    if expected_convergence:
        assert summary["termination"] == "converged"
        assert summary["stages"][-1]["max_density_change"] < 0.02
        assert summary["stages"][-1]["design_change"] < 0.02
    else:
        assert summary["termination"] == "max_stage_steps"
        assert summary["state_evaluations"] == 304
    library = ScikitFEMElasticity.from_problem(p, solver="scipy")
    assert library.evaluate(out["rho"])[0] == pytest.approx(summary["C_raw"], rel=1e-12)
    library.close()
