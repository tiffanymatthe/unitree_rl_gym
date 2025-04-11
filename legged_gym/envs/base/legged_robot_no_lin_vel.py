import torch

from legged_gym.envs.base.legged_robot import LeggedRobot

class LeggedRobotNoLinVel(LeggedRobot):
    def compute_observations(self):
        """ Computes observations
        """
        s = torch.flatten(((self.last_dof_pos.get() - self.default_dof_pos) * self.obs_scales.dof_pos).permute(1, 0, 2), start_dim=1).shape
        print("OBS SHAPE", s)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    torch.flatten(((self.last_dof_pos.get() - self.default_dof_pos) * self.obs_scales.dof_pos).permute(1, 0, 2), start_dim=1),
                                    self.last_dof_vel * self.obs_scales.dof_vel,
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[3:]

        # only uncomment when play_with_plots.py. ELSE it will mess up training because of critic
        # self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,), dim=-1)
