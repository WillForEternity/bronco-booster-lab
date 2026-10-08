"""Export training checkpoints to booster_deploy's TorchScript format (same layout as Booster's k1_walk.pt).

Runs on: laptop or GPU host (needs only torch).
    python export_stage.py <checkpoint.pt> [...] --out_dir artifacts/k1_speed_ramp/export

The k1_walk player calls the module's bare `actor` (690 inputs -> 20 joint actions), so the
export holds the actor MLP and an identity normalizer, like Booster's file.

Checkpoints are read with torch.load(weights_only=True), which loads tensors and plain data but never runs code
stored in the file. The exported policy is TorchScript: only play policies from people you trust.
"""

from __future__ import annotations

import argparse
import os

import torch
from torch import nn

from k1_conventions import ACTION_DIM, ACTOR_HIDDEN_DIMS, OBS_DIM


class Exported(nn.Module):
    def __init__(self, actor: nn.Module):
        super().__init__()
        self.obs_normalizer = nn.Identity()
        self.actor = actor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.obs_normalizer(x))


def make_actor() -> nn.Sequential:
    """RSL-RL's actor MLP for this task: OBS_DIM -> ACTOR_HIDDEN_DIMS (ELU) -> ACTION_DIM."""
    dims = [OBS_DIM, *ACTOR_HIDDEN_DIMS]
    layers: list[nn.Module] = []
    for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
        layers += [nn.Linear(d_in, d_out), nn.ELU()]
    layers.append(nn.Linear(dims[-1], ACTION_DIM))
    return nn.Sequential(*layers)


def export(path: str, out_dir: str) -> str:
    """Write <out_dir>/<checkpoint name>_policy.pt and return its path."""
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    actor = make_actor()
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")})
    module = torch.jit.script(Exported(actor).eval())
    os.makedirs(out_dir, exist_ok=True)
    stem, _ = os.path.splitext(os.path.basename(path))
    out = os.path.join(out_dir, f"{stem}_policy.pt")
    module.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    for p in a.checkpoints:
        print(export(p, a.out_dir))


if __name__ == "__main__":
    main()
