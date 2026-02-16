import math
import torch


class CPGCfg:
    # Offsets
    hip_length = 0.0838
    thigh_length = 0.213
    calf_length = 0.213

    # CPG oscillator design parameters
    convergence_factor = 150
    coupling_strength = 0.0

    # CPG pattern formation design parameters
    max_step_length = 0.15
    ground_clearance_range = (0.07, 0.12)
    ground_penetration_range = (0.0, 0.02)
    robot_height_range = (0.22, 0.32)

    # CPG-RL modulated parameter ranges
    mu_range = (1.0, 2.0)
    omega_range = (0.0, 37.699111843)  # rad / s, 6 Hz

    # Fixed initialization (e.g. for deployment)
    use_fixed_initialization = False

    fixed_ground_clearance = 0.12
    fixed_ground_penetration = 0.01
    fixed_robot_height = 0.32

class CPG:
    """CPG Implementation.
    Isaac Sim joint order: (hip FL, FR, BL, BR, thigh FL, FR, BL, BR, calf FL, FR, BL, BR)
    """

    def __init__(
        self, device="cpu", num_envs: int = 1, dt: float = 1.0 / 1000.0, decimation: int = 1, config: CPGCfg = CPGCfg()
    ):
        self.device = device
        self.num_envs = num_envs
        self.decimation = decimation
        self.dt = dt

        self.cfg = config

        # Design parameters
        self._ground_clearance = torch.zeros(num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._ground_penetration = torch.zeros(num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._robot_height = torch.zeros(num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # RL modulated parameters
        self._mu = torch.zeros(num_envs, 4, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self._omega = torch.zeros(num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # Oscillator states
        self._r = torch.zeros(num_envs, 4, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self._r_dot = torch.zeros(num_envs, 4, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self._r_ddot = torch.zeros(num_envs, 4, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self._theta = torch.zeros(num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self._theta_dot = torch.zeros(num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # Trot phase bias
        self._phase_bias = torch.tensor(
            [[0, 1, 1, 0], [-1, 0, 0, -1], [-1, 0, 0, -1], [0, 1, 1, 0]],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._phase_bias = self._phase_bias * math.pi

        # For normalization
        # Assumption - r_dot is max when r_ddot == 0.0 and (mu - r) is max
        self.r_dot_max = 0.25 * self.cfg.convergence_factor * (self.cfg.mu_range[1] - self.cfg.mu_range[0])
        # Assumption - the effect of coupling on the range is minimal
        self.theta_dot_range = self.cfg.omega_range[1] - self.cfg.omega_range[0]

    def reset(self, env_ids):
        # Reset modulated CPG parameters
        self._mu[env_ids, :, :] = 0.0
        self._omega[env_ids, :] = 0.0

        # Randomize initial oscillator states
        rand = torch.empty(len(env_ids), 4, 2, device=self.device)
        self._r[env_ids, :, :] = rand.uniform_(*self.cfg.mu_range)

        # Randomize one theta and initialize others to emulate trot gait pattern
        init_theta = rand[:, 0, 0].uniform_(0 * math.pi, 2 * math.pi)
        self._theta[env_ids, :] = init_theta.unsqueeze(1) + self._phase_bias[0, :]
        self._theta[env_ids, :] = torch.remainder(self._theta[env_ids, :], math.tau)

        # Set oscillator state derivatives to zero
        self._r_dot[env_ids, :, :] = 0.0
        self._r_ddot[env_ids, :, :] = 0.0
        self._theta_dot[env_ids, :] = 0.0

        # Randomize ground clearance, penetration, and robot height
        self._ground_clearance[env_ids, :] = rand[:, 0, 0].uniform_(*self.cfg.ground_clearance_range).unsqueeze(1)
        self._ground_penetration[env_ids, :] = rand[:, 0, 0].uniform_(*self.cfg.ground_penetration_range).unsqueeze(1)
        self._robot_height[env_ids, :] = rand[:, 0, 0].uniform_(*self.cfg.robot_height_range).unsqueeze(1)

        if self.cfg.use_fixed_initialization:
            self._r[env_ids, :, :] = 0.5 * (self.cfg.mu_range[0] + self.cfg.mu_range[1])
            self._theta[env_ids, :] = 0.5 * math.pi + self._phase_bias[0, :]
            self._theta[env_ids, :] = torch.remainder(self._theta[env_ids, :], math.tau)

            self._ground_clearance[env_ids] = self.cfg.fixed_ground_clearance
            self._ground_penetration[env_ids] = self.cfg.fixed_ground_penetration
            self._robot_height[env_ids] = self.cfg.fixed_robot_height

    def map_foot_positions(self, actions):
        # Update oscillator states
        self.update_oscillator_states(actions)

        # Map oscillator states to foot positions - pattern formation layer
        fr = 2 * (self._r - self.cfg.mu_range[0]) / (self.cfg.mu_range[1] - self.cfg.mu_range[0]) - 1

        x = -self.cfg.max_step_length * fr[:, :, 0] * torch.cos(self._theta)
        y = self.cfg.max_step_length * fr[:, :, 1] * torch.cos(self._theta)

        ground_distance = torch.where(torch.sin(self._theta) > 0, self._ground_clearance, self._ground_penetration)
        z = -self._robot_height + ground_distance * torch.sin(self._theta)

        # Incorporate offsets
        y[:, (0, 2)] -= self.cfg.hip_length
        y[:, (1, 3)] += self.cfg.hip_length

        return torch.stack((x, y, z), dim=2)

    def update_oscillator_states(self, actions):
        # Detach to prevent graph leak
        self._r = self._r.detach()
        self._r_dot = self._r_dot.detach()
        self._r_ddot = self._r_ddot.detach()
        self._theta = self._theta.detach()

        # Get modulated CPG parameters from actions
        actions_clipped = torch.clip(actions.clone(), -1, 1)
        self._mu[:, :, 0] = self.scale_actions(actions_clipped[:, 0:4], *self.cfg.mu_range)
        self._mu[:, :, 1] = self.scale_actions(actions_clipped[:, 4:8], *self.cfg.mu_range)
        self._omega = self.scale_actions(actions_clipped[:, 8:12], *self.cfg.omega_range)

        a = self.cfg.convergence_factor

        # Update oscillator states
        for _ in range(self.decimation):
            self._r_ddot = a * (0.25 * a * (self._mu - self._r) - self._r_dot)
            self._r_dot = self._r_dot + self._r_ddot * self.dt
            self._r = self._r + self._r_dot * self.dt

            # Clone to avoid evil memory sharing
            self._theta_dot = self._omega.clone()

            # Coupling
            if self.cfg.coupling_strength > 1e-6:
                r_l1_norm = torch.sum(self._r, dim=-1) * self.cfg.coupling_strength
                for i in range(4):
                    self._theta_dot[:, i] += 0.5 * torch.sum(r_l1_norm * torch.sin(self._theta - self._theta[:, i].unsqueeze(1) - self._phase_bias[i, :]), dim=-1)

            self._theta = self._theta + self._theta_dot * self.dt

            # Clip r and theta
            self._r = torch.clip(self._r, *self.cfg.mu_range)
            self._theta = torch.remainder(self._theta, math.tau)

    def compute_inverse_kinematics(self, foot_positions):
        # From SpotMicro: https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py

        hip_len = self.cfg.hip_length
        thigh_len = self.cfg.thigh_length
        calf_len = self.cfg.calf_length

        x = foot_positions[:, :, 0]
        y = foot_positions[:, :, 1]
        z = foot_positions[:, :, 2]

        D = (y**2 + (-z) ** 2 - hip_len**2 + (-x) ** 2 - thigh_len**2 - calf_len**2) / (2 * calf_len * thigh_len)
        D = torch.clamp(D, -1.0, 1.0)

        side_sign = torch.ones_like(x)
        side_sign[:, (0, 2)] *= -1

        calf_angle = torch.atan2(-torch.sqrt(1 - D**2), D)
        sqrt_component = y**2 + (-z) ** 2 - hip_len**2

        sqrt_component = torch.where(sqrt_component < 0.0, 0.0, sqrt_component)

        hip_angle = -torch.atan2(z, y) - torch.atan2(torch.sqrt(sqrt_component), side_sign * hip_len)
        thigh_angle = torch.atan2(-x, torch.sqrt(sqrt_component)) - torch.atan2(
            calf_len * torch.sin(calf_angle), thigh_len + calf_len * torch.cos(calf_angle)
        )

        return torch.stack([-hip_angle, thigh_angle, calf_angle], dim=2)

    def compute_forward_kinematics(self, joint_angles):
        hip_len = self.cfg.hip_length
        thigh_len = self.cfg.thigh_length
        calf_len = self.cfg.calf_length

        hip = joint_angles[:, :, 0]
        thigh = joint_angles[:, :, 1]
        calf = joint_angles[:, :, 2]

        sin_hip = torch.sin(hip)
        cos_hip = torch.cos(hip)
        sin_thigh = torch.sin(thigh)
        cos_thigh = torch.cos(thigh)
        sin_thigh_calf = torch.sin(thigh + calf)
        cos_thigh_calf = torch.cos(thigh + calf)

        x = -calf_len * sin_thigh_calf - thigh_len * sin_thigh

        side_sign = torch.ones_like(x)
        side_sign[:, (0, 2)] *= -1

        y = calf_len * sin_hip * cos_thigh_calf + thigh_len * sin_hip * cos_thigh + side_sign * hip_len * cos_hip
        z = -calf_len * cos_hip * cos_thigh_calf - thigh_len * cos_hip * cos_thigh + side_sign * hip_len * sin_hip

        return torch.stack((x, y, z), dim=2)

    def scale_actions(self, x, lower_lim, upper_lim):
        scaled_x = lower_lim + 0.5 * (x + 1.0) * (upper_lim - lower_lim)
        return torch.clip(scaled_x, lower_lim, upper_lim)

    def get_cpg_states(self, normalize=False):
        r = self._r.clone().reshape(-1, 8)
        theta = self._theta.clone()

        r_dot = self._r_dot.clone().reshape(-1, 8)
        theta_dot = self._theta_dot.clone()

        if normalize:
            r = 2.0 * (r - self.cfg.mu_range[0]) / (self.cfg.mu_range[1] - self.cfg.mu_range[0]) - 1
            r_dot = r_dot / self.r_dot_max
            theta_dot = 2 * theta_dot / self.theta_dot_range - 1.0

        return torch.cat((r, torch.sin(theta), torch.cos(theta), r_dot, theta_dot), dim=1)

    def get_cpg_design_params(self):
        gc = 2.0 * (self._ground_clearance - self.cfg.ground_clearance_range[0]) / (self.cfg.ground_clearance_range[1] - self.cfg.ground_clearance_range[0]) - 1
        gp = 2.0 * (self._ground_penetration - self.cfg.ground_penetration_range[0]) / (self.cfg.ground_penetration_range[1] - self.cfg.ground_penetration_range[0]) - 1
        h = 2.0 * (self._robot_height - self.cfg.robot_height_range[0]) / (self.cfg.robot_height_range[1] - self.cfg.robot_height_range[0]) - 1

        return torch.cat((gc, gp, h), dim=1)
