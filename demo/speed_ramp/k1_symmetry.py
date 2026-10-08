"""Left-right mirror of K1's k1_walk observations and actions, for symmetry-augmented PPO.

Runs on: GPU host (RSL-RL imports it by name: "k1_symmetry:compute_symmetric_states"); laptop for the unit tests.

Method: Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies" (ICRA
2024), as implemented by RSL-RL's symmetry module (`symmetry_cfg.data_augmentation_func`).

Mirroring across the robot's sagittal (x-z) plane:
- angular velocity (wx, wy, wz) -> (-wx, wy, -wz); projected gravity and linear velocity flip y;
- the command (vx, vy, wz) -> (vx, -vy, -wz);
- left and right joints swap; roll and yaw joints (x and z axes, identical on both sides in the
  MJCF) also change sign, pitch joints do not (checked against the MJCF joint ranges, e.g.
  left_hip_roll [-0.375, 1.536] vs right_hip_roll [-1.536, 0.375]);
- actions: k1_walk's player shifts the right elbow-pitch action by -0.2 before scaling, so the
  mirrored actions carry that shift across (left gets -0.2, right gets +0.2). This keeps the
  mirrored joint targets exactly mirrored, and mirroring twice returns the original.
"""

from __future__ import annotations

import torch

from k1_conventions import ELBOW_FIX_JOINT, ELBOW_FIX_OFFSET, FRAME, HISTORY, OBS_DIM, POLICY_JOINTS

N = len(POLICY_JOINTS)
PERM = [i + 1 if i % 2 == 0 else i - 1 for i in range(N)]  # pairs are stored left, right
SIGN = [-1.0 if ("roll" in j or "yaw" in j) else 1.0 for j in POLICY_JOINTS]
R_ELBOW = POLICY_JOINTS.index(ELBOW_FIX_JOINT)
L_ELBOW = POLICY_JOINTS.index(ELBOW_FIX_JOINT.replace("right", "left"))


def mirror_joints(x: torch.Tensor) -> torch.Tensor:
    """x[..., N] in POLICY_JOINTS order (positions relative to default, or velocities)."""
    return x[..., PERM] * x.new_tensor(SIGN)


def mirror_actions(a: torch.Tensor) -> torch.Tensor:
    m = mirror_joints(a)
    m[..., L_ELBOW] -= ELBOW_FIX_OFFSET
    m[..., R_ELBOW] += ELBOW_FIX_OFFSET
    return m


def mirror_frame(f: torch.Tensor) -> torch.Tensor:
    """f[..., 69]: ang vel(3), gravity(3), command(3), joint pos(20), joint vel x0.1(20), last action(20)."""
    out = torch.empty_like(f)
    out[..., 0:3] = f[..., 0:3] * f.new_tensor([-1.0, 1.0, -1.0])
    out[..., 3:6] = f[..., 3:6] * f.new_tensor([1.0, -1.0, 1.0])
    out[..., 6:9] = f[..., 6:9] * f.new_tensor([1.0, -1.0, -1.0])
    out[..., 9:9 + N] = mirror_joints(f[..., 9:9 + N])
    out[..., 9 + N:9 + 2 * N] = mirror_joints(f[..., 9 + N:9 + 2 * N])
    out[..., 9 + 2 * N:] = mirror_actions(f[..., 9 + 2 * N:])
    return out


def mirror_history(h: torch.Tensor) -> torch.Tensor:
    """h[..., 690] = 10 frames of 69, oldest first."""
    shape = h.shape
    return mirror_frame(h.reshape(*shape[:-1], HISTORY, FRAME)).reshape(shape)


def mirror_critic(c: torch.Tensor) -> torch.Tensor:
    """c[..., 693] = history (690) + base linear velocity in the body frame (3)."""
    out = torch.empty_like(c)
    out[..., :OBS_DIM] = mirror_history(c[..., :OBS_DIM])
    out[..., OBS_DIM:] = c[..., OBS_DIM:] * c.new_tensor([1.0, -1.0, 1.0])
    return out


@torch.no_grad()
def compute_symmetric_states(env=None, obs=None, actions=None):
    """RSL-RL data_augmentation_func: returns [original; left-right mirrored] along the batch."""
    obs_aug = None
    if obs is not None:
        b = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][b:] = mirror_history(obs["policy"])
        if "critic" in obs.keys():
            obs_aug["critic"][b:] = mirror_critic(obs["critic"])
    act_aug = None
    if actions is not None:
        act_aug = torch.cat([actions, mirror_actions(actions)], dim=0)
    return obs_aug, act_aug
