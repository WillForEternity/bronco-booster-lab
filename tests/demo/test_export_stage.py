"""Checkpoint export to booster_deploy's TorchScript format (demo/speed_ramp/export_stage.py). Runs on: laptop."""

import pickle

import k1_conventions as C
import pytest
import torch
from export_stage import export, make_actor

# booster_deploy's player loads TorchScript, so the export stays TorchScript although torch now deprecates it.
pytestmark = pytest.mark.filterwarnings(r"ignore:`torch\.jit\.\w+` is deprecated:FutureWarning")


def fake_checkpoint(path, actor):
    """A checkpoint shaped like RSL-RL's OnPolicyRunner.save()."""
    critic = {f"critic.{k}": torch.zeros(1) for k in ("0.weight", "0.bias")}
    state = {f"actor.{k}": v for k, v in actor.state_dict().items()} | critic | {"std": torch.ones(C.ACTION_DIM)}
    optimizer = torch.optim.Adam(actor.parameters()).state_dict()
    torch.save({"model_state_dict": state, "optimizer_state_dict": optimizer, "iter": 2500, "infos": None}, path)


def test_export_matches_the_training_actor(tmp_path):
    torch.manual_seed(0)
    actor = make_actor()
    fake_checkpoint(tmp_path / "stage_05_vmax2.00_it2500.pt", actor)
    out = export(str(tmp_path / "stage_05_vmax2.00_it2500.pt"), str(tmp_path / "export"))
    assert out.endswith("export/stage_05_vmax2.00_it2500_policy.pt")
    module = torch.jit.load(out)
    x = torch.randn(16, C.OBS_DIM)
    with torch.no_grad():
        expected = actor(x)
        assert module(x).shape == (16, C.ACTION_DIM)
        assert torch.allclose(module(x), expected)
        assert torch.allclose(module.actor(x), expected)  # booster_deploy calls the bare actor


def test_export_refuses_a_checkpoint_that_would_run_code(tmp_path):
    class Payload:
        def __reduce__(self):
            return (print, ("this must not run",))

    path = tmp_path / "evil.pt"
    with open(path, "wb") as f:
        pickle.dump({"model_state_dict": {}, "payload": Payload()}, f, protocol=2)  # torch.save's protocol
    with pytest.raises(pickle.UnpicklingError):
        export(str(path), str(tmp_path))
