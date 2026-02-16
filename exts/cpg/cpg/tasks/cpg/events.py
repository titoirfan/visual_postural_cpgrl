from __future__ import annotations
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .unitree_a1_env import UnitreeA1Env


def resample_velocity_commands(
    env: UnitreeA1Env,
    env_ids: torch.Tensor,
):
    env.sample_new_commands(env_ids)
    # That's all folks

def push_by_setting_velocity_local(
    env: UnitreeA1Env,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    push_vel_b = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)
    push_vel_w = math_utils.quat_rotate(asset.data.root_quat_w[env_ids], push_vel_b)

    # apply push in world frame    
    vel_w = asset.data.root_vel_w[env_ids]
    vel_w[:, :3] += push_vel_w

    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

def push_velocity_curriculum(
    env: UnitreeA1Env,
    env_ids: torch.Tensor,
    push_magnitudes: list[float],
    total_num_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leak_probability: float = 0.05,
    review_probability: float = 0.3,
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # determine curriculum stage
    steps_per_stage = total_num_steps / len(push_magnitudes)
    stage_idx = int(env.common_step_counter // steps_per_stage)
    stage_idx = max(0, min(len(push_magnitudes) - 1, stage_idx))

    # sample push magnitudes and directions
    cur_target = push_magnitudes[stage_idx]
    prev_target = push_magnitudes[stage_idx - 1] if stage_idx > 0 else 0.0
    min_mag = torch.full((len(env_ids),), prev_target, device=asset.device)
    max_mag = torch.full((len(env_ids),), cur_target, device=asset.device)

    # sample pushes from hardest stage
    is_leak = torch.rand(len(env_ids), device=asset.device) < leak_probability
    min_mag[is_leak] = push_magnitudes[-2] if len(push_magnitudes) > 1 else 0.0
    max_mag[is_leak] = push_magnitudes[-1]

    # sample pushes from easiest stage
    is_review = torch.rand(len(env_ids), device=asset.device) < review_probability
    min_mag[is_review] = 0.0
    max_mag[is_review] = push_magnitudes[0]

    # calculate pushes in body frame
    mags = torch.rand(len(env_ids), device=asset.device) * (max_mag - min_mag) + min_mag
    angles = torch.empty(len(env_ids), device=asset.device).uniform_(0, 2 * torch.pi)

    push_vel_b = torch.zeros(len(env_ids), 3, device=asset.device)
    push_vel_b[:, 0] = mags * torch.cos(angles)
    push_vel_b[:, 1] = mags * torch.sin(angles)

    # apply pushes in world frame
    push_vel_w = math_utils.quat_rotate(asset.data.root_quat_w[env_ids], push_vel_b)
    vel_w = asset.data.root_vel_w[env_ids]
    vel_w[:, :3] += push_vel_w

    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
