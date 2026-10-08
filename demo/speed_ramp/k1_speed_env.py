"""K1 speed task (Isaac Lab): fine-tune Booster's k1_walk policy to run. The version-3 recipe (RECIPE.md).

Runs on: GPU host (Isaac Lab 2.3.2), imported by train_v3.py and check_parity.py.

Observations, actions, gains and torque limits follow booster_deploy's k1_walk player (k1_conventions.py), so a
policy trained here plays unchanged in booster_deploy's MuJoCo player:
- policy input: the last 10 frames of 69 numbers, oldest first (690); on reset the first frame is repeated;
- joint target = default + 0.25 x action (right elbow-pitch action shifted by -0.2), low-pass filtered with
  factor 0.8 once per policy step; the head is held at its default;
- 50 Hz policy, deploy PD gains and torque limits, Booster's K1 motor model on the legs and feet.

The recipe's choices, and why (RECIPE.md has the numbered notes):
- Reward: velocity tracking plus a mechanical-power penalty. Fu et al., "Minimizing Energy Consumption Leads to the
  Emergence of Gaits in Legged Robots" (CoRL 2021): tracking plus energy alone produces natural gaits. Plus
  legged_gym's foot air time, foot slide, joint-limit, smoothness and orientation terms, and an arm-deviation
  penalty (note 1).
- Arm and head PD simulated implicitly, with lower damping on two arm joints (note 2, damping profile "v3").
- Speed curriculum: per-robot targets in [0, v_max], half of them at the frontier v_max; train_v3.py raises v_max.
- Randomization: motor command delay 0-40 ms, foot friction 0.4-1.2, motor gains x0.8-1.2, trunk mass +/-1 kg,
  pushes up to 0.4 m/s per axis.
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

from k1_conventions import (
    ACTION_FILTER,
    ACTION_SCALE,
    ACTOR_HIDDEN_DIMS,
    CLIP,
    DAMPING,
    DEFAULT_POS,
    DEPLOY_JOINTS,
    DOF_VEL_SCALE,
    EFFORT,
    ELBOW_FIX_JOINT,
    ELBOW_FIX_OFFSET,
    FRAME,
    HISTORY,
    POLICY_JOINTS,
    STIFFNESS,
    damping_overrides,
)

TASK_ID = "Bronco-K1-SpeedRamp-v0"
COMMAND_DELAY_STEPS = (0, 8)  # motor command delay, in 5 ms physics steps: 0-40 ms
IMPLICIT_ACTUATORS = ("arms", "head")  # recipe note 2: explicit PD at 5 ms chatters on the arm pitch joints


# --- Robot ------------------------------------------------------------------------------------------------------


def _armature_by_joint(act, joints: list[str]) -> dict[str, float]:
    """Booster's armature for each joint (a regex-keyed dict or one number in Booster's config)."""
    arm = act.armature if isinstance(act.armature, dict) else {j: act.armature for j in joints}
    return {j: next(v for k, v in arm.items() if re.fullmatch(k, j)) for j in joints}


def k1_robot_cfg() -> ArticulationCfg:
    """Booster's K1 asset and motor model with k1_walk's deploy gains and torque limits (recipe, "Arm PD").

    Legs and feet keep Booster's delayed PD motor model (torque-speed curve). Arms and head use implicit PD, which
    PhysX keeps stable at any time step, with the same command delay. Damping follows profile "v3".
    """
    missing = set(IMPLICIT_ACTUATORS) - set(BOOSTER_K1_CFG.actuators)
    if missing:
        raise KeyError(f"Booster's K1 config has no actuator group {sorted(missing)}; check the booster_train pin")
    damping = dict(DAMPING, **damping_overrides(DEPLOY_JOINTS, "v3"))
    actuators = {}
    for name, act in BOOSTER_K1_CFG.actuators.items():
        joints = [j for j in DEPLOY_JOINTS if any(re.fullmatch(e, j) for e in act.joint_names_expr)]
        stiffness = {j: STIFFNESS[j] for j in joints}
        damp = {j: damping[j] for j in joints}
        effort = {j: EFFORT[j] for j in joints}
        min_delay, max_delay = COMMAND_DELAY_STEPS
        if name in IMPLICIT_ACTUATORS:
            actuators[name] = DelayedImplicitActuatorCfg(
                joint_names_expr=joints, stiffness=stiffness, damping=damp, effort_limit_sim=effort,
                armature=_armature_by_joint(act, joints), min_delay=min_delay, max_delay=max_delay)
        else:
            new = act.replace(stiffness=stiffness, damping=damp)
            new.effort_limit = effort
            new.effort_limit_sim = effort
            new.min_delay, new.max_delay = min_delay, max_delay
            actuators[name] = new
    return BOOSTER_K1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.57), joint_pos=dict(DEFAULT_POS), joint_vel={".*": 0.0}),
        actuators=actuators,
    )


def forward_speed(robot) -> torch.Tensor:
    """Forward speed in the heading frame (m/s), ignoring pitch and roll."""
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[:, 0]


# --- Command: the speed curriculum ------------------------------------------------------------------------------


class SpeedCurriculumCommand(CommandTerm):
    """Per-robot forward-speed targets in [v_min, v_max]; a share of robots is held at the frontier
    [v_max - frontier_width, v_max]. Targets are redrawn mid-episode, so the policy keeps slower speeds and learns
    to change speed. train_v3.py raises v_max.

    Also accumulates frontier tracking and how episodes ended (fall or time-out), for train_v3.py's Isaac gate.
    """

    cfg: SpeedCurriculumCommandCfg

    def __init__(self, cfg: SpeedCurriculumCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.v_max = float(cfg.v_max_start)
        self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.since = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._acc = {k: torch.zeros((), device=self.device)
                     for k in ("ratio_sum", "ratio_n", "f_falls", "f_done", "a_falls", "a_done")}

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command

    def set_v_max(self, v_max: float) -> None:
        self.v_max = float(v_max)

    def _frontier(self, target: torch.Tensor) -> torch.Tensor:
        return target >= self.v_max - self.cfg.frontier_width - 1e-6

    def pop_stats(self) -> dict:
        """Sums since the last call: frontier speed ratio, and falls / finished episodes (frontier and all)."""
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


# --- Action: k1_walk's joint-target convention ------------------------------------------------------------------


class K1WalkAction(ActionTerm):
    cfg: K1WalkActionCfg

    def __init__(self, cfg: K1WalkActionCfg, env: ManagerBasedRLEnv):
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


# --- Observation: k1_walk's 10-frame history --------------------------------------------------------------------


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


# --- Rewards ----------------------------------------------------------------------------------------------------


def mechanical_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum over joints of |applied torque x joint velocity| (W)."""
    asset = env.scene[asset_cfg.name]
    return (asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]).abs().sum(dim=1)


FEET = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")


@configclass
class RewardsCfg:
    track_lin_vel = RewTerm(func=loco_mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0,
                            params={"command_name": "base_velocity", "std": 0.5})
    track_heading = RewTerm(func=loco_mdp.track_ang_vel_z_world_exp, weight=0.5,
                            params={"command_name": "base_velocity", "std": 0.5})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # Weight from measurement: with implicit arm PD (note 2), k1_walk draws 131 W at 1.0 m/s and 248 W at
    # 1.5 m/s, so -7e-4 is about 9-17% of the full tracking reward.
    energy = RewTerm(func=mechanical_power, weight=-7.0e-4)
    feet_air_time = RewTerm(func=loco_mdp.feet_air_time_positive_biped, weight=0.5,
                            params={"command_name": "base_velocity", "sensor_cfg": FEET, "threshold": 0.4})
    feet_slide = RewTerm(func=loco_mdp.feet_slide, weight=-0.1,
                         params={"sensor_cfg": FEET, "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    hip_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])})
    # Note 1: without it the first candidate spent 251 W in the arms in MuJoCo (k1_walk: 4 W) and missed the speed
    # bar. -0.1 is Isaac Lab's G1 value; moderate arm swing remains.
    arm_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7,
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*"])})


# --- Environment ------------------------------------------------------------------------------------------------


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
    robot: ArticulationCfg = k1_robot_cfg()
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class CommandsCfg:
    base_velocity = SpeedCurriculumCommandCfg()


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
    # Term order is kept as trained: it fixes the order in which the randomizations draw random numbers.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
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
    trunk_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk"), "mass_distribution_params": (-1.0, 1.0),
                "operation": "add"},
    )
    motor_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2), "operation": "scale"},
    )
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},  # harder than eval_mujoco.py's 0.3 m/s push
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
    rewards: RewardsCfg = RewardsCfg()
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


def make_env_cfg(num_envs: int) -> K1SpeedEnvCfg:
    cfg = K1SpeedEnvCfg()
    cfg.scene.num_envs = num_envs
    return cfg


@configclass
class K1SpeedPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000  # train_v3.py stops earlier by the recipe's stopping rule (note 6)
    save_interval = 250
    experiment_name = "k1_run_v3"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.25,  # small: start close to k1_walk's own actions
        actor_hidden_dims=list(ACTOR_HIDDEN_DIMS),
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
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:K1SpeedEnvCfg", "rsl_rl_cfg_entry_point": f"{__name__}:K1SpeedPPOCfg"},
)
