"""Speed measures used by eval_mujoco.py. Pure Python: no MuJoCo or booster_deploy import, so the unit tests can load it.

Runs on: GPU host or laptop (imported by eval_mujoco.py), and by the unit tests.
"""

from __future__ import annotations

import math


def heading_speed(d) -> float:
    """Trunk velocity along its heading (m/s). A free joint's qvel[0:3] is the world-frame linear velocity."""
    w, x, y, z = d.qpos[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(d.qvel[0] * math.cos(yaw) + d.qvel[1] * math.sin(yaw))


def epte_sp(speeds: list[float], command: float, total_steps: int) -> float:
    """EPTE-SP (Li, Li and Hutter, RA-L 2026, Eq. 8, as described in RECIPE.md, note 5).

    `speeds` are the per-policy-step heading speeds up to the fall (or the whole trial). Each step's
    error is |v - command| / command; every planned step after a fall counts as 1.0 (100%).
    """
    errs = [abs(v - command) / command for v in speeds]
    return (sum(errs) + max(0, total_steps - len(errs))) / total_steps
