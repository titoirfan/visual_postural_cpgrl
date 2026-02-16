import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from modules.cpg import CPGCfg
from modules.reflex import ReflexCfg

from .events import resample_velocity_commands, push_by_setting_velocity_local, push_velocity_curriculum
from .terrains import ROUGH_TERRAINS_CFG, STAIRS_TERRAINS_CFG
from .terrains import EVAL_FLAT_TERRAINS_CFG, EVAL_DISCRETE_TERRAINS_CFG, EVAL_ROUGH_TERRAINS_CFG, EVAL_WAVE_TERRAINS_CFG, EVAL_STAIRS_TERRAINS_CFG

from .unitree_a1_env_cfg import UnitreeA1FlatEnvCfg


@configclass
class CPGUnitreeA1FlatEnvCfg(UnitreeA1FlatEnvCfg):
    # Action space is CPG parameters - mu, omega, psi
    action_space = 12
    action_scale = 0.5

    # Observation space - 45 + foot contact states (4) + rx, ry, sin theta, cos theta, rx_dot, ry_dot, omega (28)
    observation_space = 77

    # CPG config
    cpg_config: CPGCfg = CPGCfg()
    observe_cpg_design_parameters = True
    if observe_cpg_design_parameters:
        observation_space += 3

    # reward scales
    lin_vel_reward_scale = 2.0
    yaw_rate_reward_scale = 1.0

    # penalty scales
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_power_reward_scale = -0.001

    joint_torque_reward_scale = 0.0
    joint_accel_reward_scale = 0.0
    feet_air_time_reward_scale = 0.0
    action_rate_reward_scale = 0.0
    undesired_contact_reward_scale = 0.0
    flat_orientation_reward_scale = 0.0

    # Visualization
    debug_vis = True

    # Reflexes and exteroception
    enable_reflex_network = False
    enable_exteroception = False

    reflex_config: ReflexCfg = ReflexCfg()

    # Height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.9, 0.6)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    if enable_reflex_network:
        action_space += 12
        observation_space += 12

    if enable_exteroception:
        observation_space += 70
    state_space = observation_space

    def __post_init__(self):
        super().__post_init__()

        # Simulation config
        self.sim.dt = 1.0 / 1000.0
        self.decimation = 10
        self.sim.render_interval = self.decimation

        # Match sensor update periods
        self.contact_sensor.update_period = self.sim.dt
        self.height_scanner.update_period = self.decimation * self.sim.dt

        # Commands
        # These commands are higher than Visual CPG-RL's to allow the robot cross the large curriculum levels
        self.commands.lin_vel_x_ranges = (-0.6, 0.6)
        self.commands.lin_vel_y_ranges = (-0.6, 0.6)

        # Lower robot spawn location (previously 0.42)
        self.robot.init_state.pos = (0.0, 0.0, 0.35)

        # Increase the robot's joint PD gains
        self.robot.actuators["base_legs"].stiffness = 100.0
        self.robot.actuators["base_legs"].damping = 2.0

        # Events
        # On startup
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.0)

        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.scale_link_masses = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )

        # Interval-based
        self.events.push_robot = EventTerm(
            func=push_velocity_curriculum,
            mode="interval",
            interval_range_s=(8, 12),
            params={
                "push_magnitudes": [0.25, 0.5, 0.75, 1.0],  # m/s
                "total_num_steps": 48 * 800,  # num_steps_per_env * max_iterations
            },
        )

        self.events.resample_commands = EventTerm(
            func=resample_velocity_commands,
            mode="interval",
            interval_range_s=(8, 8),
            params={},
        )


@configclass
class CPGUnitreeA1RoughEnvCfg(CPGUnitreeA1FlatEnvCfg):
    # curriculum
    enable_curriculum = True

    # Rough terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=ROUGH_TERRAINS_CFG.num_rows - 1,
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

    def __post_init__(self):
        super().__post_init__()


@configclass
class CPGUnitreeA1StairsEnvCfg(CPGUnitreeA1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Change terrain
        self.terrain.terrain_generator = STAIRS_TERRAINS_CFG

        # Expand height scanner
        self.height_scanner.pattern_cfg.size = (1.6, 1.0)

        # Enable exteroception and reflexes
        self.enable_exteroception = True
        self.enable_reflex_network = True

        self.action_space = 24
        self.observation_space = 77 + 3 + 12 + 187
        self.state_space = self.observation_space


@configclass
class CPGUnitreeA1FlatEnvCfg_PLAY(CPGUnitreeA1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observation_noises.enable = False

        # Disable interval-based pushes
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Disable on-reset randomizations
        self.events.add_base_mass = None
        self.events.scale_link_masses = None
        self.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 0.8)

        # Fix the CPG design parameters
        self.cpg_config.use_fixed_initialization = True


@configclass
class CPGUnitreeA1RoughEnvCfg_PLAY(CPGUnitreeA1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.num_rows = 10
            self.terrain.terrain_generator.num_cols = 10
            self.terrain.terrain_generator.curriculum = True

        self.observation_noises.enable = False

        # Disable interval-based pushes
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Disable on-reset randomizations
        self.events.add_base_mass = None
        self.events.scale_link_masses = None
        self.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 0.8)

        # Fix the CPG design parameters
        self.cpg_config.use_fixed_initialization = True


@configclass
class CPGUnitreeA1StairsEnvCfg_PLAY(CPGUnitreeA1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # Change terrain
        self.terrain.terrain_generator = STAIRS_TERRAINS_CFG

        # Expand height scanner
        self.height_scanner.pattern_cfg.size = (1.6, 1.0)

        # Enable exteroception and reflexes
        self.enable_exteroception = True
        self.enable_reflex_network = True

        self.action_space = 24
        self.observation_space = 77 + 3 + 12 + 187
        self.state_space = self.observation_space


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL(CPGUnitreeA1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 30.0
        self.seed = 42

        self.eval_mode = True
        self.eval_scheduled_velocity = False

        self.enable_curriculum = False

        # Commands
        self.commands.lin_vel_x_ranges = (0.6, 0.6)
        self.commands.lin_vel_y_ranges = (0.0, 0.0)
        self.commands.ang_vel_z_ranges = (0.0, 0.0)

        self.commands.sample_heading_tracking_envs = False
        self.commands.sample_standing_still_envs = False

        # env_spacing is unused when using TerrainGenerator
        self.scene.num_envs = 1024

        # Terrain
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=EVAL_FLAT_TERRAINS_CFG,
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

        # Disable command resampling
        self.events.resample_commands = None

        # Interval-based pushes
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Disable on-reset randomizations
        self.events.add_base_mass = None
        self.events.scale_link_masses = None
        self.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 0.8)

        # Fix the CPG design parameters
        self.cpg_config.use_fixed_initialization = True


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_IdealRough(CPGUnitreeA1RoughEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EVAL_ROUGH_TERRAINS_CFG


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_IdealDiscrete(CPGUnitreeA1RoughEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EVAL_DISCRETE_TERRAINS_CFG


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_IdealWave(CPGUnitreeA1RoughEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EVAL_WAVE_TERRAINS_CFG


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_PushFlat(CPGUnitreeA1RoughEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()

        self.events.push_robot = EventTerm(
            func=push_by_setting_velocity_local,
            mode="interval",
            interval_range_s=(5, 5),
            params={"velocity_range": {"x": (0.0, 0.0), "y": (1.0, 1.0)}},
        )


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_PushRough(CPGUnitreeA1RoughEnvCfg_EVAL_PushFlat):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EVAL_ROUGH_TERRAINS_CFG


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_PushDiscrete(CPGUnitreeA1RoughEnvCfg_EVAL_PushFlat):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EVAL_DISCRETE_TERRAINS_CFG


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_PushWave(CPGUnitreeA1RoughEnvCfg_EVAL_PushFlat):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EVAL_WAVE_TERRAINS_CFG


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_Gait(CPGUnitreeA1RoughEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 8.0


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_StabilitySideS(CPGUnitreeA1RoughEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 9.0
        self.events.push_robot = EventTerm(
            func=push_by_setting_velocity_local,
            mode="interval",
            interval_range_s=(5, 5),
            params={"velocity_range": {"x": (0.0, 0.0), "y": (0.5, 0.5)}},
        )


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_StabilitySideM(CPGUnitreeA1RoughEnvCfg_EVAL_StabilitySideS):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.params = {"velocity_range": {"x": (0.0, 0.0), "y": (1.0, 1.0)}}


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_StabilitySideL(CPGUnitreeA1RoughEnvCfg_EVAL_StabilitySideS):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.params = {"velocity_range": {"x": (0.0, 0.0), "y": (1.5, 1.5)}}


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_Tracking(CPGUnitreeA1RoughEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 24.0
        self.eval_scheduled_velocity = True

        self.terrain.terrain_generator = EVAL_ROUGH_TERRAINS_CFG
        self.events.push_robot = None


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_TrackingDiscrete(CPGUnitreeA1RoughEnvCfg_EVAL_Tracking):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EVAL_DISCRETE_TERRAINS_CFG


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_StressHeavy(CPGUnitreeA1RoughEnvCfg_EVAL_PushDiscrete):
    def __post_init__(self):
        super().__post_init__()

        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "mass_distribution_params": (6.0, 6.0),
                "operation": "add",
            },
        )


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_StressSlippery(CPGUnitreeA1RoughEnvCfg_EVAL_PushDiscrete):
    def __post_init__(self):
        super().__post_init__()

        self.events.physics_material.params["static_friction_range"] = (0.4, 0.4)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.3)


@configclass
class CPGUnitreeA1RoughEnvCfg_EVAL_StressHeavySlippery(CPGUnitreeA1RoughEnvCfg_EVAL_StressHeavy):
    def __post_init__(self):
        super().__post_init__()

        self.events.physics_material.params["static_friction_range"] = (0.4, 0.4)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.3)


@configclass
class CPGUnitreeA1StairsEnvCfg_EVAL(CPGUnitreeA1RoughEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0

        # Change terrain
        self.terrain.terrain_generator = EVAL_STAIRS_TERRAINS_CFG

        # Expand height scanner
        self.height_scanner.pattern_cfg.size = (1.6, 1.0)

        # Enable exteroception and reflexes
        self.enable_exteroception = True
        self.enable_reflex_network = True

        self.action_space = 24
        self.observation_space = 77 + 3 + 12 + 187
        self.state_space = self.observation_space


@configclass
class CPGUnitreeA1StairsEnvCfg_EVAL_Push(CPGUnitreeA1StairsEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()

        self.events.push_robot = EventTerm(
            func=push_by_setting_velocity_local,
            mode="interval",
            interval_range_s=(5, 5),
            params={"velocity_range": {"x": (0.0, 0.0), "y": (1.0, 1.0)}},
        )
