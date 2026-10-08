"""K1 and k1_walk conventions (demo/speed_ramp/k1_conventions.py). Runs on: laptop."""

import k1_conventions as C
import pytest


def test_dimensions():
    assert (C.FRAME, C.OBS_DIM, C.ACTION_DIM) == (69, 690, 20)


def test_policy_joints_are_deploy_joints_without_the_head():
    assert set(C.POLICY_JOINTS) == {j for j in C.DEPLOY_JOINTS if "head" not in j}
    assert len(set(C.POLICY_JOINTS)) == len(C.POLICY_JOINTS)


def test_every_deploy_joint_has_pose_gains_and_limits():
    for table in (C.DEFAULT_POS, C.STIFFNESS, C.DAMPING, C.EFFORT):
        assert list(table) == C.DEPLOY_JOINTS


def test_v3_damping_lowers_exactly_the_two_unstable_arm_joints():
    # Recipe note 2: elbow pitch 2.0 -> 0.7 and shoulder pitch 2.0 -> 1.0, on both sides; nothing else changes.
    assert C.damping_overrides(C.DEPLOY_JOINTS, "v3") == {
        "aaleft_shoulder_pitch_joint": 1.0, "aaright_shoulder_pitch_joint": 1.0,
        "left_elbow_pitch_joint": 0.7, "right_elbow_pitch_joint": 0.7,
    }
    assert C.damping_overrides(C.DEPLOY_JOINTS, None) == {}


def test_unknown_damping_profile_is_an_error():
    with pytest.raises(ValueError, match="unknown damping profile"):
        C.damping_overrides(C.DEPLOY_JOINTS, "v4")


def test_ambiguous_damping_rule_is_an_error(monkeypatch):
    monkeypatch.setitem(C.DAMPING_PROFILES, "test", {"elbow": 1.0, "elbow_pitch": 0.5})
    with pytest.raises(ValueError, match="more than one rule"):
        C.damping_overrides(C.DEPLOY_JOINTS, "test")
