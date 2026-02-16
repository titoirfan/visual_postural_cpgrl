"""
This script demonstrates foot trajectory generation via CPG on a Unitree A1.
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates foot trajectory generation via CPG on a Unitree A1."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils
import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from modules.cpg import CPG


def design_scene() -> Articulation:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Origin with Unitree A1
    prim_utils.create_prim("/World/Origin", "Xform", translation=torch.zeros(3))

    unitree_a1_cfg = UNITREE_A1_CFG
    unitree_a1_cfg.prim_path = "/World/Origin/Robot"
    unitree_a1_cfg.spawn.articulation_props.fix_root_link = True

    unitree_a1 = Articulation(unitree_a1_cfg)

    return unitree_a1


def run_simulator(sim: sim_utils.SimulationContext, entity: Articulation):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    count_per_reset = 1000

    cpg = CPG(device=sim.device, dt=sim_dt, decimation=1)

    cpg.cfg.omega_range = (0.0, 12.566370614)
    cpg.cfg.use_fixed_initialization = True

    # Use CPG
    mu_x = torch.ones(1, 4, device=sim.device) * 1.0
    mu_y = torch.ones(1, 4, device=sim.device) * 0.0
    omega = torch.ones(1, 4, device=sim.device) * 1.0
    actions = torch.cat((mu_x, mu_y, omega), dim=1)

    previous_body_positions = entity.data.body_pos_w

    # Simulate physics
    while simulation_app.is_running():
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        # reset periodically
        if count % count_per_reset == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset robot root state
            root_state = entity.data.default_root_state.clone()
            entity.write_root_pose_to_sim(root_state[:, :7])
            entity.write_root_velocity_to_sim(root_state[:, 7:])

            # reset robot joint state
            joint_pos, joint_vel = entity.data.default_joint_pos.clone(), entity.data.default_joint_vel.clone()
            entity.write_joint_state_to_sim(joint_pos, joint_vel)

            # reset robot internal state
            entity.reset()
            print("[INFO]: Resetting robot state...")

            # Reset CPG state
            cpg.reset(torch.tensor([0]))
            print("[INFO]: Resetting CPG state...")

            draw_interface.clear_lines()
            previous_body_positions = entity.data.body_pos_w

        # Compute CPG foot position targets and convert to joint angles via inverse kinematics
        foot_positions = cpg.map_foot_positions(actions)
        joint_angles = cpg.compute_inverse_kinematics(foot_positions)

        # Compose joint position targets
        joint_pos_target = entity.data.default_joint_pos.clone()
        joint_pos_target[0, :] = joint_angles.permute(0, 2, 1).reshape(-1, 12)

        # Plot foot trajectory
        plotted_foot_ids = entity.find_bodies(".*_foot")[0]

        line_source_positions = previous_body_positions[:, plotted_foot_ids, :].reshape(-1, 3)
        line_target_positions = entity.data.body_pos_w[:, plotted_foot_ids, :].reshape(-1, 3)

        line_colors = [[1.0, 0.0, 0.0, 1.0]] * line_source_positions.shape[0]
        line_thicknesses = [3.0] * line_source_positions.shape[0]

        draw_interface.draw_lines(line_source_positions.tolist(), line_target_positions.tolist(), line_colors, line_thicknesses)

        previous_body_positions = entity.data.body_pos_w

        # apply action to the robot
        entity.set_joint_position_target(joint_pos_target)
        # write data to sim
        entity.write_data_to_sim()

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        entity.update(sim_dt)


def main():
    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[1.0, 1.0, 0.84], target=[0.0, 0.0, 0.21])

    # Design scene
    entity = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, entity)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
