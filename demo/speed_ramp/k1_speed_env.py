"""K1 speed-ramp task: fine-tune Booster's k1_walk policy to go faster.

Runs on: GPU host (Isaac Lab 2.3.2), imported by train_v3.py and check_parity.py.

The observation and action conventions copy booster_deploy's k1_walk player (commit 7bb1462,
tasks/locomotion/locomotion.py and tasks/locomotion/robots/k1/__init__.py), so a policy trained
here plays unchanged in booster_deploy's MuJoCo player:
- 69 numbers per frame: body angular velocity, projected gravity, velocity command (x, y, yaw),
  joint positions minus default, joint velocities x 0.1, last action. Joints in POLICY_JOINTS order.
- The last 10 frames, oldest first, flattened (690 numbers). On reset the first frame is repeated.
- Joint target = default + 0.25 x action (right elbow pitch action shifted by -0.2), then
  low-pass filtered with factor 0.8 once per policy step. The head is held at its default.
- 50 Hz policy, deploy PD gains and torque limits.

Versions 1 and 2 of the task (make_env_cfg, make_env_cfg_v2: reward presets "speed_only" and
"shaped") stay for reference; train_v3.py uses make_env_cfg_v3 (RECIPE.md).
"""

from __future__ import annotations

import re

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg, ManagerTermBase
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from booster_train.assets.robots.actuator import DelayedImplicitActuatorCfg
from booster_train.assets.robots.booster_k1 import BOOSTER_K1_CFG

# --- booster_deploy k1_walk conventions -------------------------------------------------------

# booster_deploy K1 joint order (booster_deploy/robots/k1.py) with k1_walk's defaults and gains.
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

# k1_walk's policy joint order (20 joints; the head is not controlled).
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
HISTORY = 10
FRAME = 3 + 3 + 3 + 3 * len(POLICY_JOINTS)  # 69
ACTION_SCALE = 0.25
ACTION_FILTER = 0.8
ELBOW_FIX_JOINT, ELBOW_FIX_OFFSET = "right_elbow_pitch_joint", 0.2
DOF_VEL_SCALE = 0.1
CLIP = 100.0


def _k1_walk_robot_cfg(actuator_delay: bool, ideal_pd: bool = False, delay_range: tuple[int, int] | None = None,
                       implicit_upper: bool = False, damping: dict | None = None) -> ArticulationCfg:
    """Booster's K1 asset and motor model (torque-speed curve), with k1_walk's deploy gains and limits.

    ideal_pd=True replaces the motor model with plain PD (torque clipped to the deploy limits, no
    torque-speed curve, no delay), which is what booster_deploy's MuJoCo player applies. Armature
    stays Booster's (it matches the MJCF's joint armature values). It ignores the other options, so
    combining them is an error.
    """
    if ideal_pd:
        if delay_range is not None or implicit_upper or damping:
            raise ValueError("ideal_pd=True takes no delay_range, implicit_upper or damping")
        from isaaclab.actuators import IdealPDActuatorCfg

        actuators = {}
        for name, act in BOOSTER_K1_CFG.actuators.items():
            joints = [j for j in DEPLOY_JOINTS if any(re.fullmatch(e, j) for e in act.joint_names_expr)]
            arm = act.armature if isinstance(act.armature, dict) else {j: act.armature for j in joints}
            arm = {j: next(v for k, v in arm.items() if re.fullmatch(k, j)) for j in joints}
            actuators[name] = IdealPDActuatorCfg(
                joint_names_expr=joints, stiffness={j: STIFFNESS[j] for j in joints}, damping={j: DAMPING[j] for j in joints},
                effort_limit={j: EFFORT[j] for j in joints}, effort_limit_sim={j: EFFORT[j] for j in joints}, armature=arm)
        return BOOSTER_K1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.57), joint_pos=dict(DEFAULT_POS), joint_vel={".*": 0.0}),
            actuators=actuators,
        )
    actuators = {}
    for name, act in BOOSTER_K1_CFG.actuators.items():
        exprs = act.joint_names_expr
        joints = [j for j in DEPLOY_JOINTS if any(re.fullmatch(e, j) for e in exprs)]
        damp = dict(DAMPING, **(damping or {}))
        new = act.replace(stiffness={j: STIFFNESS[j] for j in joints}, damping={j: damp[j] for j in joints})
        new.effort_limit = {j: EFFORT[j] for j in joints}
        new.effort_limit_sim = {j: EFFORT[j] for j in joints}
        if not actuator_delay:
            new.min_delay, new.max_delay = 0, 0
        elif delay_range is not None:
            new.min_delay, new.max_delay = delay_range
        if implicit_upper and name in ("arms", "head"):
            # Implicit PD (PhysX): stable at any time step. Explicit PD at 5 ms chatters on the arm pitch joints
            # (damping x dt / inertia up to 6.7; stable needs < 2). Same delay model as the legs.
            arm = new.armature if isinstance(new.armature, dict) else {j: new.armature for j in joints}
            arm = {j: next(v for k, v in arm.items() if re.fullmatch(k, j)) for j in joints}
            new = DelayedImplicitActuatorCfg(
                joint_names_expr=joints, stiffness={j: STIFFNESS[j] for j in joints}, damping={j: damp[j] for j in joints},
                effort_limit_sim={j: EFFORT[j] for j in joints}, armature=arm, min_delay=new.min_delay, max_delay=new.max_delay)
        actuators[name] = new
    return BOOSTER_K1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.57), joint_pos=dict(DEFAULT_POS), joint_vel={".*": 0.0}),
        actuators=actuators,
    )


def forward_speed(robot) -> torch.Tensor:
    """Forward speed in the heading frame (m/s), ignoring pitch and roll."""
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[:, 0]


# --- Command: the target speed ----------------------------------------------------------------


class SpeedRampCommand(CommandTerm):
    """Velocity command (target, 0, 0) for every env. The target is raised from outside.

    Also accumulates the mean forward speed of robots that are past the first `settle_steps` of
    their episode, for the ramp's "target hit" check.
    """

    cfg: "SpeedRampCommandCfg"

    def __init__(self, cfg: "SpeedRampCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.target = float(cfg.start_speed)
        self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command[:, 0] = self.target
        self.speed_sum = torch.zeros((), device=self.device)
        self.speed_count = torch.zeros((), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command

    def set_target(self, speed: float) -> None:
        self.target = float(speed)
        self.vel_command[:, 0] = self.target

    def pop_mean_speed(self) -> float | None:
        n = self.speed_count.item()
        mean = self.speed_sum.item() / n if n > 0 else None
        self.speed_sum.zero_()
        self.speed_count.zero_()
        return mean

    def _update_metrics(self):
        settled = (self._env.episode_length_buf >= self.cfg.settle_steps).float()
        self.speed_sum += (forward_speed(self.robot) * settled).sum()
        self.speed_count += settled.sum()

    def _resample_command(self, env_ids):
        self.vel_command[env_ids, 0] = self.target
        self.vel_command[env_ids, 1:] = 0.0

    def _update_command(self):
        pass


@configclass
class SpeedRampCommandCfg(CommandTermCfg):
    class_type: type = SpeedRampCommand
    asset_name: str = "robot"
    start_speed: float = 1.0
    settle_steps: int = 50  # ignore the first 1 s of each episode (accelerating from standstill)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)


# --- Action: k1_walk's joint-target convention ------------------------------------------------


class K1WalkAction(ActionTerm):
    cfg: "K1WalkActionCfg"

    def __init__(self, cfg: "K1WalkActionCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._policy_ids, _ = self._asset.find_joints(POLICY_JOINTS, preserve_order=True)
        self._fix_col = POLICY_JOINTS.index(ELBOW_FIX_JOINT)
        self._raw = torch.zeros(self.num_envs, len(POLICY_JOINTS), device=self.device)
        self._default = self._asset.data.default_joint_pos.clone()
        self._filtered = self._default.clone()

    @property
    def action_dim(self) -> int:
        return len(POLICY_JOINTS)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._filtered

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions.clamp(-CLIP, CLIP)
        shifted = self._raw.clone()
        shifted[:, self._fix_col] -= ELBOW_FIX_OFFSET
        target = self._default.clone()
        target[:, self._policy_ids] += shifted * ACTION_SCALE
        self._filtered.lerp_(target, ACTION_FILTER)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._filtered)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._filtered[ids] = self._default[ids]


@configclass
class K1WalkActionCfg(ActionTermCfg):
    class_type: type = K1WalkAction
    asset_name: str = "robot"


# --- Observation: k1_walk's 10-frame history --------------------------------------------------


class K1WalkHistory(ManagerTermBase):
    """690 numbers: the last 10 frames of k1_walk's 69-number observation, oldest first."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.joint_ids, _ = self.robot.find_joints(POLICY_JOINTS, preserve_order=True)
        self.hist = torch.zeros(env.num_envs, HISTORY, FRAME, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.fresh[:] = True
        else:
            self.fresh[env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        d = self.robot.data
        frame = torch.cat(
            [
                d.root_ang_vel_b,
                d.projected_gravity_b,
                env.command_manager.get_command("base_velocity"),
                d.joint_pos[:, self.joint_ids] - d.default_joint_pos[:, self.joint_ids],
                d.joint_vel[:, self.joint_ids] * DOF_VEL_SCALE,
                env.action_manager.get_term("joint_pos").raw_actions,
            ],
            dim=-1,
        ).clamp(-CLIP, CLIP)
        self.hist = torch.roll(self.hist, shifts=-1, dims=1)
        self.hist[:, -1] = frame
        if self.fresh.any():
            self.hist[self.fresh] = frame[self.fresh].unsqueeze(1).expand(-1, HISTORY, -1)
            self.fresh[:] = False
        return self.hist.reshape(env.num_envs, -1)


# --- Rewards ----------------------------------------------------------------------------------


def track_forward_speed_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    target = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-((forward_speed(env.scene["robot"]) - target) ** 2) / std**2)


FEET = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")


@configclass
class SpeedOnlyRewards:
    """Match the target speed; falling ends the episode with a penalty. Nothing else."""

    track_speed = RewTerm(func=track_forward_speed_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class ShapedRewards(SpeedOnlyRewards):
    """Speed plus the usual Isaac Lab humanoid penalties (adapted from the G1 velocity task)."""

    track_heading = RewTerm(
        func=loco_mdp.track_ang_vel_z_world_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    feet_air_time = RewTerm(
        func=loco_mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={"command_name": "base_velocity", "sensor_cfg": FEET, "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=loco_mdp.feet_slide,
        weight=-0.1,
        params={"sensor_cfg": FEET, "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    ankle_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*"])}
    )
    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    arm_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-1.25e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*"])}
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])},
    )


REWARD_PRESETS = {"speed_only": SpeedOnlyRewards, "shaped": ShapedRewards}


# --- Environment ------------------------------------------------------------------------------


@configclass
class K1SpeedSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
        ),
    )
    robot: ArticulationCfg = _k1_walk_robot_cfg(actuator_delay=True)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class CommandsCfg:
    base_velocity = SpeedRampCommandCfg()


@configclass
class ActionsCfg:
    joint_pos = K1WalkActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        history = ObsTerm(func=K1WalkHistory)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        history = ObsTerm(func=K1WalkHistory)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale, mode="reset", params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    trunk_contact = DoneTerm(
        func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk|trunk"), "threshold": 1.0}
    )
    tipped_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.3})


@configclass
class K1SpeedEnvCfg(ManagerBasedRLEnvCfg):
    scene: K1SpeedSceneCfg = K1SpeedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: ShapedRewards = ShapedRewards()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


def make_env_cfg(reward_preset: str, num_envs: int, actuator_delay: bool = True, ideal_pd: bool = False,
                 friction: float | None = None) -> K1SpeedEnvCfg:
    """friction=None keeps the randomized foot friction; a number fixes static and dynamic friction to it."""
    cfg = K1SpeedEnvCfg()
    cfg.rewards = REWARD_PRESETS[reward_preset]()
    cfg.scene.num_envs = num_envs
    cfg.scene.robot = _k1_walk_robot_cfg(actuator_delay, ideal_pd)
    if friction is not None:
        cfg.events.physics_material.params["static_friction_range"] = (friction, friction)
        cfg.events.physics_material.params["dynamic_friction_range"] = (friction, friction)
    return cfg


@configclass
class K1SpeedPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000  # the ramp script decides when to stop
    save_interval = 250
    experiment_name = "k1_speed_ramp"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.25,  # small: start close to k1_walk's own actions
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,  # k1_walk was trained without one; deploy calls the bare actor
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


gym.register(
    id="Bronco-K1-SpeedRamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:K1SpeedEnvCfg", "rsl_rl_cfg_entry_point": f"{__name__}:K1SpeedPPOCfg"},
)


# ==============================================================================================
# Version 2 (October 7, 2026, evening): speed-range curriculum and varied training conditions.
# Version 1 above is kept unchanged so its runs stay reproducible.
#
# Changes, following standard practice for learned fast locomotion (Margolis et al., "Rapid
# Locomotion via Reinforcement Learning", RSS 2022; Rudin et al., legged_gym, CoRL 2021):
# - Each robot gets its own target speed, drawn from [v_min, v_max]; a share of robots is held at
#   the frontier [v_max - frontier_width, v_max]. Targets are redrawn mid-episode, so the policy
#   keeps slower speeds and learns to change speed.
# - The command term records frontier tracking and how episodes ended (fall vs time-out), so the
#   training script can promote v_max only when the frontier is tracked well AND rarely falls.
# - Training conditions vary: random pushes, trunk mass, and motor gains, on top of friction and
#   the motor model's command delay.
# ==============================================================================================


class SpeedCurriculumCommand(CommandTerm):
    cfg: "SpeedCurriculumCommandCfg"

    def __init__(self, cfg: "SpeedCurriculumCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.v_max = float(cfg.v_max_start)
        self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.since = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        z = lambda: torch.zeros((), device=self.device)  # noqa: E731
        self._acc = {k: z() for k in ("ratio_sum", "ratio_n", "f_falls", "f_done", "a_falls", "a_done")}

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command

    def set_v_max(self, v_max: float) -> None:
        self.v_max = float(v_max)

    def _frontier(self, target: torch.Tensor) -> torch.Tensor:
        return target >= self.v_max - self.cfg.frontier_width - 1e-6

    def pop_stats(self) -> dict:
        out = {k: v.item() for k, v in self._acc.items()}
        for v in self._acc.values():
            v.zero_()
        return out

    def reset(self, env_ids=None):
        # Record how finished episodes ended before resampling. The termination buffers still hold
        # this step's values here (ManagerBasedRLEnv resets the command manager before them).
        if env_ids is not None and not isinstance(env_ids, slice):
            ids = torch.as_tensor(env_ids, device=self.device)
            valid = (self._env.episode_length_buf[ids] > 0).float()  # skip the initial reset
            fell = self._env.termination_manager.terminated[ids].float() * valid
            frontier = self._frontier(self.vel_command[ids, 0]).float() * valid
            self._acc["a_falls"] += fell.sum()
            self._acc["a_done"] += valid.sum()
            self._acc["f_falls"] += (fell * frontier).sum()
            self._acc["f_done"] += frontier.sum()
        return super().reset(env_ids)

    def _update_metrics(self):
        self.since += 1
        target = self.vel_command[:, 0]
        ok = ((self.since >= self.cfg.settle_steps) & self._frontier(target) & (target > 0.1)).float()
        ratio = forward_speed(self.robot) / target.clamp(min=0.1)
        self._acc["ratio_sum"] += (ratio * ok).sum()
        self._acc["ratio_n"] += ok.sum()

    def _resample_command(self, env_ids):
        n = len(env_ids)
        lo = max(self.cfg.v_min, self.v_max - self.cfg.frontier_width)
        at_frontier = torch.rand(n, device=self.device) < self.cfg.frontier_frac
        u = torch.rand(n, device=self.device)
        target = torch.where(at_frontier, lo + (self.v_max - lo) * u, self.cfg.v_min + (self.v_max - self.cfg.v_min) * u)
        self.vel_command[env_ids, 0] = target
        self.vel_command[env_ids, 1:] = 0.0
        self.since[env_ids] = 0

    def _update_command(self):
        pass


@configclass
class SpeedCurriculumCommandCfg(CommandTermCfg):
    class_type: type = SpeedCurriculumCommand
    asset_name: str = "robot"
    v_max_start: float = 1.0
    v_min: float = 0.0
    frontier_width: float = 0.25
    frontier_frac: float = 0.5
    settle_steps: int = 50  # ignore 1 s after each new target when measuring tracking
    resampling_time_range: tuple[float, float] = (10.0, 10.0)


@configclass
class CommandsCfgV2:
    base_velocity = SpeedCurriculumCommandCfg()


@configclass
class EventsCfgV2(EventsCfg):
    trunk_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk"), "mass_distribution_params": (-1.0, 1.0),
                "operation": "add"},
    )
    motor_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1), "operation": "scale"},
    )
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},  # harder than eval_mujoco.py's 0.3 m/s push
    )


def make_env_cfg_v2(reward_preset: str, num_envs: int) -> K1SpeedEnvCfg:
    cfg = make_env_cfg(reward_preset, num_envs, actuator_delay=True)
    cfg.commands = CommandsCfgV2()
    cfg.events = EventsCfgV2()
    return cfg



# ==============================================================================================
# Version 3 (October 7, 2026, evening): the single training recipe for a fast, natural run.
# Version 2's speed-range curriculum and stability gate are kept (train_v3.py adds a MuJoCo gate).
#
# - Reward: velocity tracking + mechanical-power penalty. Fu et al., "Minimizing Energy Consumption
#   Leads to the Emergence of Gaits in Legged Robots" (CoRL 2021): tracking plus energy alone
#   produces natural gaits and gait transitions. Plus legged_gym's foot air time, foot slide,
#   joint-limit, smoothness and orientation terms. The arm-deviation penalty was first removed so
#   the arms could swing, then restored (recipe note 1; see RewardsV3.arm_deviation).
# - Arm and head PD simulated implicitly, with lower damping on two arm joints (recipe note 2; see
#   V3_DAMPING).
# - Symmetry: mirrored-data augmentation in PPO (k1_symmetry.py; Mittal et al., ICRA 2024).
# - Wider randomization, centered on the calibration of October 7: motor command delay 0-40 ms
#   (k1_walk's fine-tuned children slow down when the delay is removed), foot friction 0.4-1.2,
#   motor gains x0.8-1.2, trunk mass +/-1 kg, pushes up to 0.4 m/s per axis.
# ==============================================================================================


def mechanical_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum over joints of |applied torque x joint velocity| (W)."""
    asset = env.scene[asset_cfg.name]
    return (asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]).abs().sum(dim=1)


@configclass
class RewardsV3:
    track_lin_vel = RewTerm(func=loco_mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0,
                            params={"command_name": "base_velocity", "std": 0.5})
    track_heading = RewTerm(func=loco_mdp.track_ang_vel_z_world_exp, weight=0.5,
                            params={"command_name": "base_velocity", "std": 0.5})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # Weight from measurement. Before recipe note 2's fix, arm chatter inflated power to ~500-600 W; with implicit arm PD
    # k1_walk draws 131 W at 1.0 m/s and 248 W at 1.5 m/s, so -7e-4 is ~9-17% of the full tracking reward.
    energy = RewTerm(func=mechanical_power, weight=-7.0e-4)
    feet_air_time = RewTerm(func=loco_mdp.feet_air_time_positive_biped, weight=0.5,
                            params={"command_name": "base_velocity", "sensor_cfg": FEET, "threshold": 0.4})
    feet_slide = RewTerm(func=loco_mdp.feet_slide, weight=-0.1,
                         params={"sensor_cfg": FEET, "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    hip_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])})
    # Restored (recipe note 1): without it the first candidate spent
    # 251 W in the arms in MuJoCo (k1_walk: 4 W) and missed the speed bar. -0.1 is Isaac Lab's G1 value; moderate
    # swing remains.
    arm_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7,
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*"])})


# Recipe note 2: arm PD. Explicit PD is stable only if
# damping x dt / inertia < 2. In booster_deploy's MuJoCo player (2 ms) the elbow-pitch inertia falls to 0.00148 kg m^2 in some postures, so damping 2.0 gives 2.69:
# a 250 Hz chatter at the torque limit (124 W per elbow). Shoulder pitch reaches 1.94. Isaac's explicit PD at 5 ms
# chatters on all arm pitch joints. Version 3 therefore (a) simulates arm and head PD implicitly in Isaac, and
# (b) lowers damping only on the two unstable joints (eval_mujoco.DAMPING_PROFILES["v3"] uses the same values).
V3_DAMPING = {"aaleft_shoulder_pitch_joint": 1.0, "aaright_shoulder_pitch_joint": 1.0,
              "left_elbow_pitch_joint": 0.7, "right_elbow_pitch_joint": 0.7}


def make_env_cfg_v3(num_envs: int) -> K1SpeedEnvCfg:
    cfg = K1SpeedEnvCfg()
    cfg.rewards = RewardsV3()
    cfg.scene.num_envs = num_envs
    cfg.scene.robot = _k1_walk_robot_cfg(actuator_delay=True, delay_range=(0, 8),  # 0-40 ms at 5 ms physics steps
                                         implicit_upper=True, damping=V3_DAMPING)
    cfg.commands = CommandsCfgV2()
    cfg.events = EventsCfgV2()
    cfg.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.2)
    cfg.events.motor_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
    cfg.events.motor_gains.params["damping_distribution_params"] = (0.8, 1.2)
    return cfg
