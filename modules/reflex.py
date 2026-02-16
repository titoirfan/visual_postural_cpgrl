import torch


class ReflexCfg:
    # Oscillator design parameters
    convergence_factor = 150

    # Foot position adjustment ranges (m)
    q_x_range = (-0.10, 0.10)
    q_y_range = (-0.10, 0.10)
    q_z_range = (-0.05, 0.05)

    # Whether to directly correct joint positions instead of foot positions
    joint_pos_correction_mode = True

    # Joint angle adjustment ranges (rad)
    q_hip_range = (-0.25, 0.25)
    q_thigh_range = (-0.25, 0.25)
    q_calf_range = (-0.25, 0.25)


class Reflex:
    def __init__(
        self,
        device="cpu",
        num_envs: int = 1,
        dt: float = 1.0 / 1000.0,
        decimation: int = 1,
        config: ReflexCfg = ReflexCfg(),
    ):
        self.device = device
        self.num_envs = num_envs
        self.decimation = decimation
        self.dt = dt

        self.cfg = config

        # Oscillator states
        self._q = torch.zeros(num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._q_dot = torch.zeros(num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._q_ddot = torch.zeros(num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def reset(self, env_ids):
        # Reset oscillator states
        self._q[env_ids, :, :] = 0.0
        self._q_dot[env_ids, :, :] = 0.0
        self._q_ddot[env_ids, :, :] = 0.0

    def get_adjustments(self, actions):
        # Detach to prevent graph leak
        self._q = self._q.detach()
        self._q_dot = self._q_dot.detach()
        self._q_ddot = self._q_ddot.detach()

        # Get target foot position adjustments
        actions_clipped = torch.clip(actions.clone(), -1, 1).reshape(-1, 4, 3)

        q_desired = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.joint_pos_correction_mode:
            q_desired[:, :, 0] = self.scale_actions(actions_clipped[:, :, 0], *self.cfg.q_hip_range)
            q_desired[:, :, 1] = self.scale_actions(actions_clipped[:, :, 1], *self.cfg.q_thigh_range)
            q_desired[:, :, 2] = self.scale_actions(actions_clipped[:, :, 2], *self.cfg.q_calf_range)
        else:
            q_desired[:, :, 0] = self.scale_actions(actions_clipped[:, :, 0], *self.cfg.q_x_range)
            q_desired[:, :, 1] = self.scale_actions(actions_clipped[:, :, 1], *self.cfg.q_y_range)
            q_desired[:, :, 2] = self.scale_actions(actions_clipped[:, :, 2], *self.cfg.q_z_range)

        a = self.cfg.convergence_factor

        # Update oscillator states
        for _ in range(self.decimation):
            self._q_ddot = a * (0.25 * a * (q_desired - self._q) - self._q_dot)
            self._q_dot = self._q_dot + self._q_ddot * self.dt
            self._q = self._q + self._q_dot * self.dt

        return self._q.clone()

    def scale_actions(self, x, lower_lim, upper_lim):
        scaled_x = lower_lim + 0.5 * (x + 1.0) * (upper_lim - lower_lim)
        return torch.clip(scaled_x, lower_lim, upper_lim)
