"""Left-right mirror for symmetry-augmented PPO (demo/speed_ramp/k1_symmetry.py). Runs on: laptop."""

import ast
from pathlib import Path

import k1_symmetry as S
import torch

ENV_FILE = Path(__file__).resolve().parents[2] / "demo" / "speed_ramp" / "k1_speed_env.py"


def env_constant(name: str):
    """Read a literal constant from k1_speed_env.py without importing it (it needs Isaac Lab)."""
    for node in ast.parse(ENV_FILE.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if getattr(target, "id", None) == name:
            return ast.literal_eval(node.value)
        if isinstance(target, ast.Tuple):  # A, B = 1, 2
            ids = [getattr(e, "id", None) for e in target.elts]
            if name in ids:
                return ast.literal_eval(node.value)[ids.index(name)]
    raise KeyError(name)


def test_joint_order_matches_the_training_task():
    # k1_symmetry keeps its own copy so RSL-RL can import it alone; it must not drift from the task's.
    assert env_constant("POLICY_JOINTS") == S.POLICY_JOINTS
    assert env_constant("HISTORY") == S.HISTORY
    assert env_constant("ELBOW_FIX_JOINT") == "right_elbow_pitch_joint"
    assert env_constant("ELBOW_FIX_OFFSET") == S.ELBOW_SHIFT


def test_pairs_are_left_then_right():
    for i in range(0, S.N, 2):
        left, right = S.POLICY_JOINTS[i], S.POLICY_JOINTS[i + 1]
        assert "left" in left and "right" in right
        assert left.replace("left", "") == right.replace("right", "")


def test_mirroring_twice_is_identity():
    torch.manual_seed(0)
    obs = torch.randn(64, S.HISTORY * S.FRAME)
    critic = torch.randn(64, S.HISTORY * S.FRAME + 3)
    act = torch.randn(64, S.N)
    assert torch.allclose(S.mirror_history(S.mirror_history(obs)), obs, atol=1e-6)
    assert torch.allclose(S.mirror_critic(S.mirror_critic(critic)), critic, atol=1e-6)
    assert torch.allclose(S.mirror_actions(S.mirror_actions(act)), act, atol=1e-6)


def joint_targets(actions: torch.Tensor) -> torch.Tensor:
    """k1_walk's convention relative to the default pose: 0.25 x action, right elbow-pitch action shifted by -0.2."""
    shifted = actions.clone()
    shifted[..., S.R_ELBOW] -= S.ELBOW_SHIFT
    return 0.25 * shifted


def test_mirrored_actions_give_mirrored_joint_targets():
    # Default poses are mirror images (checked below), so mirrored offsets from default = mirrored targets.
    torch.manual_seed(1)
    act = torch.randn(32, S.N)
    assert torch.allclose(joint_targets(S.mirror_actions(act)), S.mirror_joints(joint_targets(act)), atol=1e-6)


def env_tables() -> dict:
    """Evaluate k1_speed_env.py's joint tables (plain expressions over literals) without importing Isaac Lab."""
    names = {"DEPLOY_JOINTS", "_LEG", "DEFAULT_POS", "STIFFNESS", "DAMPING", "EFFORT"}
    nodes = [n for n in ast.parse(ENV_FILE.read_text()).body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in names]
    ns: dict = {}
    exec(compile(ast.Module(nodes, []), str(ENV_FILE), "exec"), ns)
    return ns


def test_default_pose_and_gains_are_mirror_symmetric():
    t = env_tables()
    pose = torch.tensor([t["DEFAULT_POS"][j] for j in S.POLICY_JOINTS])
    assert torch.allclose(S.mirror_joints(pose), pose)
    for table in ("STIFFNESS", "DAMPING", "EFFORT"):
        gains = torch.tensor([t[table][j] for j in S.POLICY_JOINTS])
        assert torch.equal(gains[S.PERM], gains), table


def test_frame_layout_signs():
    f = torch.zeros(S.FRAME)
    f[0:9] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    m = S.mirror_frame(f)
    assert m[0:9].tolist() == [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0, -9.0]


def test_augmentation_doubles_the_batch():
    act = torch.randn(8, S.N)
    _, aug = S.compute_symmetric_states(actions=act)
    assert aug.shape == (16, S.N)
    assert torch.equal(aug[:8], act)
