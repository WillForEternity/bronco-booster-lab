"""Our MuJoCo scripts against Booster's real player. Opt-in: needs booster_deploy (pinned commit) and its environment.

Runs on: laptop, from this repository (about a minute):
    BOOSTER_DEPLOY=../booster_deploy .venv/bin/pytest tests/integration
Skipped when BOOSTER_DEPLOY is not set, as in CI.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = os.environ.get("BOOSTER_DEPLOY")
pytestmark = pytest.mark.skipif(not DEPLOY, reason="set BOOSTER_DEPLOY to booster_deploy's folder (with its .venv)")

SCRIPTS = Path(__file__).resolve().parents[2] / "demo" / "speed_ramp"


def deploy_python() -> str:
    venv = Path(DEPLOY).resolve() / ".venv"
    return str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))


def run(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(SCRIPTS), PYTHONWARNINGS="ignore")
    return subprocess.run([deploy_python(), *args], cwd=cwd or DEPLOY, env=env, capture_output=True, text=True,
                          timeout=600)


def test_k1_walk_survives_perturbed_trials(tmp_path):
    out = tmp_path / "result.json"
    p = run([str(SCRIPTS / "eval_mujoco.py"), "--speed", "1.0", "--trials", "2", "--seconds", "4", "--out", str(out)])
    assert p.returncode == 0, p.stderr
    r = json.loads(out.read_text())
    assert r["survived"] == 2 and len(r["per_trial"]) == 2
    assert 0.8 <= r["mean_forward_speed_survivors_mps"] <= 1.2  # k1_walk walks at about 0.97 m/s here
    assert json.loads(p.stdout)["survived"] == 2  # stdout carries the summary


def test_v3_damping_reaches_the_player():
    code = ("import json\n"
            "from booster_player import k1_walk_cfg\n"
            "base, v3 = k1_walk_cfg(), k1_walk_cfg(damping_profile='v3')\n"
            "print(json.dumps({n: [float(a), float(b)] for n, a, b in zip(base.robot.joint_names,"
            " base.robot.joint_damping, v3.robot.joint_damping) if a != b}))")
    p = run(["-c", code])
    assert p.returncode == 0, p.stderr
    changed = json.loads(p.stdout.strip().splitlines()[-1])
    assert changed == {"aaleft_shoulder_pitch_joint": [2.0, 1.0], "left_elbow_pitch_joint": [2.0, 0.7],
                       "aaright_shoulder_pitch_joint": [2.0, 1.0], "right_elbow_pitch_joint": [2.0, 0.7]}


def test_record_stage_writes_a_video_and_metrics(tmp_path):
    out = tmp_path / "clip.mp4"
    p = run([str(SCRIPTS / "record_stage.py"), "--speed", "1.0", "--seconds", "1", "--out", str(out)])
    assert p.returncode == 0, p.stderr
    m = json.loads(out.with_suffix(".json").read_text())
    assert out.stat().st_size > 0 and m["safety_stop_s"] is None and len(m["checkpoint_sha256"]) == 64


def test_wrong_working_directory_is_explained(tmp_path):
    p = run([str(SCRIPTS / "eval_mujoco.py"), "--speed", "1.0"], cwd=str(tmp_path))
    assert p.returncode != 0 and "Run this from booster_deploy's repository root" in p.stderr
