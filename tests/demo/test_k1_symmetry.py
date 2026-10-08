"""Left-right mirror for symmetry-augmented PPO (demo/speed_ramp/k1_symmetry.py). Runs on: laptop."""

import k1_conventions as C
import k1_symmetry as S
import torch


def test_pairs_are_left_then_right():
    for i in range(0, S.N, 2):
        left, right = C.POLICY_JOINTS[i], C.POLICY_JOINTS[i + 1]
        assert "left" in left and "right" in right
        assert left.replace("left", "") == right.replace("right", "")


def test_mirroring_twice_is_identity():
    torch.manual_seed(0)
    obs = torch.randn(64, C.OBS_DIM)
    critic = torch.randn(64, C.OBS_DIM + 3)
    act = torch.randn(64, S.N)
    assert torch.allclose(S.mirror_history(S.mirror_history(obs)), obs, atol=1e-6)
    assert torch.allclose(S.mirror_critic(S.mirror_critic(critic)), critic, atol=1e-6)
    assert torch.allclose(S.mirror_actions(S.mirror_actions(act)), act, atol=1e-6)


def joint_targets(actions: torch.Tensor) -> torch.Tensor:
    """k1_walk's convention relative to the default pose: scale x action, right elbow-pitch action shifted first."""
    shifted = actions.clone()
    shifted[..., S.R_ELBOW] -= C.ELBOW_FIX_OFFSET
    return C.ACTION_SCALE * shifted


def test_mirrored_actions_give_mirrored_joint_targets():
    # Default poses are mirror images (checked below), so mirrored offsets from default = mirrored targets.
    torch.manual_seed(1)
    act = torch.randn(32, S.N)
    assert torch.allclose(joint_targets(S.mirror_actions(act)), S.mirror_joints(joint_targets(act)), atol=1e-6)


def test_default_pose_and_gains_are_mirror_symmetric():
    pose = torch.tensor([C.DEFAULT_POS[j] for j in C.POLICY_JOINTS])
    assert torch.allclose(S.mirror_joints(pose), pose)
    for name, table in (("STIFFNESS", C.STIFFNESS), ("DAMPING", C.DAMPING), ("EFFORT", C.EFFORT)):
        gains = torch.tensor([table[j] for j in C.POLICY_JOINTS])
        assert torch.equal(gains[S.PERM], gains), name


def test_v3_damping_is_mirror_symmetric():
    overrides = C.damping_overrides(C.POLICY_JOINTS, "v3")
    damping = torch.tensor([overrides.get(j, C.DAMPING[j]) for j in C.POLICY_JOINTS])
    assert torch.equal(damping[S.PERM], damping)


def test_frame_layout_signs():
    f = torch.zeros(C.FRAME)
    f[0:9] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    m = S.mirror_frame(f)
    assert m[0:9].tolist() == [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0, -9.0]


def test_augmentation_doubles_the_batch():
    act = torch.randn(8, S.N)
    _, aug = S.compute_symmetric_states(actions=act)
    assert aug.shape == (16, S.N)
    assert torch.equal(aug[:8], act)
