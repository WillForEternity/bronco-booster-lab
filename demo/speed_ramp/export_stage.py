"""Export ramp checkpoints to booster_deploy's TorchScript format (same layout as k1_walk.pt).

Runs on: laptop or GPU host (needs only torch).
    python export_stage.py <stage_checkpoint.pt> [...] --out_dir artifacts/k1_speed_ramp/<run>

The k1_walk player calls the module's bare `actor` (690 inputs -> 20 joint actions), so the
export holds the actor MLP and an identity normalizer, like Booster's file.
"""

import argparse
import os

import torch
from torch import nn

OBS, ACT = 690, 20


class Exported(nn.Module):
    def __init__(self, actor: nn.Module):
        super().__init__()
        self.obs_normalizer = nn.Identity()
        self.actor = actor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.obs_normalizer(x))


def export(path: str, out_dir: str) -> str:
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    actor = nn.Sequential(nn.Linear(OBS, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                          nn.Linear(128, ACT))
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")})
    module = torch.jit.script(Exported(actor).eval())
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(path).replace(".pt", "_policy.pt"))
    module.save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    for p in a.checkpoints:
        print(export(p, a.out_dir))
