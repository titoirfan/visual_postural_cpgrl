from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.sensors import RayCaster

from modules.cpg import CPG
from modules.reflex import Reflex

from .cpg_env_cfg import CPGUnitreeA1FlatEnvCfg, CPGUnitreeA1RoughEnvCfg, CPGUnitreeA1StairsEnvCfg
from .unitree_a1_env import UnitreeA1Env


class CPGUnitreeA1Env(UnitreeA1Env):
    cfg: CPGUnitreeA1FlatEnvCfg | CPGUnitreeA1RoughEnvCfg | CPGUnitreeA1StairsEnvCfg

    def __init__(
        self,
        cfg: CPGUnitreeA1FlatEnvCfg | CPGUnitreeA1RoughEnvCfg | CPGUnitreeA1StairsEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg = cfg
        self.cpg = CPG(
            device=self.device,
            num_envs=self.num_envs,
            dt=cfg.sim.dt,
            decimation=cfg.decimation,
            config=self.cfg.cpg_config,
        )

        # Actions and previous actions are CPG parameters - mu, omega, psi
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # CPG pattern formation layer variables
        self._foot_position_targets = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._joint_position_targets = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # Reflex
        self.reflex = None
        if self.cfg.enable_reflex_network:
            self.reflex = Reflex(
                device=self.device,
                num_envs=self.num_envs,
                dt=cfg.sim.dt,
                decimation=cfg.decimation,
                config=self.cfg.reflex_config,
            )

        # Logging
        self._episode_sums.update({
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "joint_power",
            ]
        })

    def _setup_scene(self):
        # Add a height scanner for perceptive locomotion
        if self.cfg.enable_exteroception:
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner

        return super()._setup_scene()

    def _pre_physics_step(self, actions):
        # This function is only called once per step (same rate as policy)
        self._actions = actions.clone()
        self._processed_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device)

        cpg_actions = self._actions[:, :12]
        self._foot_position_targets = self.cpg.map_foot_positions(cpg_actions * self.cfg.action_scale)

        if self.cfg.enable_reflex_network:
            reflex_actions = self._actions[:, 12:]
            if self.cfg.reflex_config.joint_pos_correction_mode:
                self._processed_actions = self.reflex.get_adjustments(reflex_actions * self.cfg.action_scale).permute(0, 2, 1).reshape(-1, 12)
            else:
                self._foot_position_targets = self._foot_position_targets + self.reflex.get_adjustments(reflex_actions * self.cfg.action_scale)

        self._joint_position_targets = self.cpg.compute_inverse_kinematics(self._foot_position_targets)
        self._processed_actions = self._processed_actions + self._joint_position_targets.permute(0, 2, 1).reshape(-1, 12)

    def _reset_idx(self, env_ids):
        # Reset CPG states
        self.cpg.reset(env_ids)

        # Reset reflex network
        if self.cfg.enable_reflex_network:
            self.reflex.reset(env_ids)

        # Reset and update log
        super()._reset_idx(env_ids)

    def _get_observations(self):
        self._update_commands()
        if self.cfg.eval_mode and self.cfg.eval_scheduled_velocity:
            self._get_scheduled_commands()

        self._previous_actions = self._actions.clone()

        height_data = None
        if self.cfg.enable_exteroception:
            height_data = self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5

        # torch.max() returns values and indices, [0] is for accessing the values
        foot_contact_states = torch.max(torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids, :], dim=-1), dim=1)[0] > 1.0
        foot_contact_states = foot_contact_states.float()

        sensory_obs = {
            "base_ang_vel": self._robot.data.root_ang_vel_b.clone(),
            "base_projected_gravity": self._robot.data.projected_gravity_b.clone(),
            "joint_pos": self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            "joint_vel": self._robot.data.joint_vel - self._robot.data.default_joint_vel,
            "foot_contact_states": foot_contact_states,
            "height_scan": height_data,
        }
        sensory_state = {key: value.clone() if value is not None else None for key, value in sensory_obs.items()}

        if self.cfg.observation_noises.enable:
            self._apply_observation_noises(sensory_obs)

        # Scale and clip policy sensory obs
        if sensory_obs["height_scan"] is not None:
            sensory_obs["height_scan"] = sensory_obs["height_scan"].clip(-1.0, 1.0)

        # Scale and clip noiseless sensory obs for critic
        if sensory_state["height_scan"] is not None:
            sensory_state["height_scan"] = sensory_state["height_scan"].clip(-1.0, 1.0)

        obs = torch.cat([tensor for tensor in sensory_obs.values() if tensor is not None] + [self._commands, self._actions, self.cpg.get_cpg_states()], dim=-1)
        critic_obs = torch.cat([tensor for tensor in sensory_state.values() if tensor is not None] + [self._commands, self._actions, self.cpg.get_cpg_states()], dim=-1)

        if self.cfg.observe_cpg_design_parameters:
            obs = torch.cat((obs, self.cpg.get_cpg_design_params()), dim=-1)
            critic_obs = torch.cat((critic_obs, self.cpg.get_cpg_design_params()), dim=-1)

        observations = {"policy": obs, "critic": critic_obs}

        return observations

    def _get_rewards(self):
        reward = super()._get_rewards()

        # Joint power
        joint_power = torch.sum(torch.abs(self._robot.data.applied_torque * self._robot.data.joint_vel), dim=1)

        extra_rewards = {
            "joint_power": joint_power * self.cfg.joint_power_reward_scale * self.step_dt,
        }
        extra_reward = torch.sum(torch.stack(list(extra_rewards.values())), dim=0)

        for key, value in extra_rewards.items():
            self._episode_sums[key] += value

        return extra_reward + reward
