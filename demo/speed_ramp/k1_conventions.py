"""K1 and k1_walk conventions shared by training, export, symmetry and playback. The single source for these values.

Runs on: GPU host or laptop. Pure Python (no torch, Isaac Lab or booster_deploy), so every script and the unit
tests can import it.

The values copy booster_deploy's k1_walk player (commit 7bb1462: booster_deploy/robots/k1.py,
tasks/locomotion/locomotion.py and tasks/locomotion/robots/k1/__init__.py), so a policy trained with them plays
unchanged in booster_deploy's MuJoCo player.
"""

from __future__ import annotations

# booster_deploy's K1 joint order, with k1_walk's default pose, PD gains and torque limits.
DEPLOY_JOINTS = [
    "aahead_yaw_joint", "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
_LEG = [-0.15, 0.0, 0.0, 0.3, -0.15, 0.0]
DEFAULT_POS = dict(zip(DEPLOY_JOINTS, [0, 0, 0.2, -1.25, 0, -0.5, 0.2, 1.25, 0, 0.5, *_LEG, *_LEG], strict=True))
STIFFNESS = dict(zip(DEPLOY_JOINTS, [4.0] * 2 + [20.0] * 8 + [100.0, 100.0, 100.0, 100.0, 65.0, 65.0] * 2, strict=True))
DAMPING = dict(zip(DEPLOY_JOINTS, [1.0] * 2 + [2.0] * 8 + [2.0, 2.0, 2.0, 2.0, 1.0, 1.0] * 2, strict=True))
EFFORT = dict(zip(DEPLOY_JOINTS, [6.0] * 2 + [14.0] * 8 + [30.0, 20.0, 15.0, 35.0, 24.0, 15.0] * 2, strict=True))

# k1_walk's policy joint order: 20 joints, stored as (left, right) pairs. The head is not controlled.
POLICY_JOINTS = [
    "aaleft_shoulder_pitch_joint", "aaright_shoulder_pitch_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
    "left_knee_pitch_joint", "right_knee_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

# Observation: per frame, body angular velocity (3), projected gravity (3), velocity command (3), joint positions
# minus default, joint velocities x DOF_VEL_SCALE, last action (20 each). The last HISTORY frames, oldest first.
HISTORY = 10
FRAME = 9 + 3 * len(POLICY_JOINTS)  # 69
OBS_DIM = HISTORY * FRAME  # 690
ACTION_DIM = len(POLICY_JOINTS)  # 20
DOF_VEL_SCALE = 0.1
CLIP = 100.0

# Action: joint target = default + ACTION_SCALE x action, with the right elbow-pitch action shifted by
# -ELBOW_FIX_OFFSET first, then low-pass filtered with ACTION_FILTER once per policy step.
ACTION_SCALE = 0.25
ACTION_FILTER = 0.8
ELBOW_FIX_JOINT, ELBOW_FIX_OFFSET = "right_elbow_pitch_joint", 0.2

# Actor network (ELU MLP), as trained by RSL-RL and exported for booster_deploy.
ACTOR_HIDDEN_DIMS = [512, 256, 128]

# Joint damping profiles (recipe note 2), keyed by a substring of the joint name. "v3": the two arm joints whose
# explicit PD is unstable at MuJoCo's 2 ms step in some postures (damping x dt / inertia: elbow pitch 2.69,
# shoulder pitch 1.94; stable needs < 2) are lowered, giving a worst-case ratio of about 0.95. Every other joint
# keeps k1_walk's damping. Policies from train_v3.py are trained and played with "v3".
DAMPING_PROFILES = {"v3": {"shoulder_pitch": 1.0, "elbow_pitch": 0.7}}


def damping_overrides(joint_names: list[str], profile: str | None) -> dict[str, float]:
    """The damping a profile sets, by joint name ({} for None). Joints the profile does not name are left out."""
    if not profile:
        return {}
    if profile not in DAMPING_PROFILES:
        raise ValueError(f"unknown damping profile {profile!r}; known: {sorted(DAMPING_PROFILES)}")
    rules = DAMPING_PROFILES[profile]
    out = {}
    for name in joint_names:
        matches = [kd for key, kd in rules.items() if key in name]
        if len(matches) > 1:
            raise ValueError(f"joint {name!r} matches more than one rule of damping profile {profile!r}")
        if matches:
            out[name] = matches[0]
    return out
