# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    # on startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    scale_link_masses = None
    startup_pd_gains = None

    # on reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = None
    resample_commands = None


@configclass
class CommandsCfg:
    # command velocity ranges
    lin_vel_x_ranges = (-1.0, 1.0)
    lin_vel_y_ranges = (-1.0, 1.0)
    ang_vel_z_ranges = (-1.0, 1.0)

    # randomly sample envs to be given stand still commands
    sample_standing_still_envs = True
    standing_still_envs_prob = 0.02

    # randomly sample envs to be given heading tracking commands instead of constant angular velocity
    sample_heading_tracking_envs = True
    heading_tracking_envs_prob = 1.0
    heading_target_ranges = (-math.pi, math.pi)
    heading_tracking_kp = 0.5


@configclass
class ObservationNoisesCfg:
    enable = True
    noises = {
        "base_lin_vel": UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        "base_ang_vel": UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        "base_projected_gravity": UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        "joint_pos": UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        "joint_vel": UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        "height_scan": UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    }


@configclass
class UnitreeA1FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 45
    state_space = observation_space

    # eval mode
    eval_mode = False
    save_eval_logs = False
    eval_scheduled_velocity = False

    # commands
    commands: CommandsCfg = CommandsCfg()

    # curriculum - irrelevant for flat env
    enable_curriculum = False

    # observation noises
    observation_noises: ObservationNoisesCfg = ObservationNoisesCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True
    )

    # lights
    light = sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    )

    # reward scales
    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.75
    feet_air_time_reward_scale = 0.25

    # penalty scales
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2e-4
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    undesired_contact_reward_scale = 0.0
    flat_orientation_reward_scale = -2.5

    # Visualization
    debug_vis = True

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    def __post_init__(self):
        pass


@configclass
class UnitreeA1RoughEnvCfg(UnitreeA1FlatEnvCfg):
    # env
    observation_space = 232
    state_space = observation_space

    # curriculum
    enable_curriculum = True

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    terrain.terrain_generator.curriculum = enable_curriculum

    # scale down the terrains because the robot is small
    terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
    terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
    terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
    terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.025, 0.1)
    terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.1)

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    feet_air_time_reward_scale = 0.01
    flat_orientation_reward_scale = 0.0

    # Visualization
    debug_vis = True

    def __post_init__(self):
        pass


@configclass
class UnitreeA1FlatEnvCfg_PLAY(UnitreeA1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observation_noises.enable = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeA1RoughEnvCfg_PLAY(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.num_rows = 5
            self.terrain.terrain_generator.num_cols = 5
            self.terrain.terrain_generator.curriculum = False

        self.observation_noises.enable = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
