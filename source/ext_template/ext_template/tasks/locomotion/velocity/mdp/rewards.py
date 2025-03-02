from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


from torch import distributions
# @torch.jit.script
# def quat_rotate_inverse(q, v):
#     shape = q.shape
#     q_w = q[:, -1]
#     q_vec = q[:, :3]
#     a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
#     b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
#     c = q_vec * \
#         torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
#             shape[0], 3, 1)).squeeze(-1) * 2.0
#     return a - b + c







def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward




def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # asset_feet_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # asset_feet: Articulation = env.scene[asset_feet_cfg.name]
    # feet_positions = asset_feet.data.body_pos_w[:, asset_feet_cfg.body_ids, :]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = (target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1))
        # print("sensor pos: ",sensor.data.pos_w[:, 2].unsqueeze(1)[0])
        # print("ray hits pos: ",torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)[0])
        # print("body pos: ",asset.data.root_pos_w[:, 2][0])
        # print("feet pos: ",feet_positions[0])
        # print((asset.data.root_pos_w[:, 2] - adjusted_target_height)[0])


        # print("next")
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    # print(abs(asset.data.root_pos_w[:, 2] - adjusted_target_height))
    return torch.square(abs(asset.data.root_pos_w[:, 2] - adjusted_target_height).clamp(min=0.0, max=1.0))


def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)

def feet_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize if the distance between feet is below a minimum threshold."""

    asset: Articulation = env.scene[asset_cfg.name]

    feet_position_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, 0:2] # (num_envs, num_feet, 2)

    if feet_position_xy is None:
        return torch.zeros(env.num_envs)

    # feet distance on x-y plane
    feet_distance_front = torch.norm(feet_position_xy[:, 0, :2] - feet_position_xy[:, 1, :2], dim=-1)
    feet_distance_behind = torch.norm(feet_position_xy[:, 2, :2] - feet_position_xy[:, 3, :2], dim=-1)
    feet_distance = torch.norm(torch.stack([feet_distance_front, feet_distance_behind], dim=1), dim=-1)  # (num_envs, 1)

    return torch.clamp(0.45 - feet_distance, min=0.0)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward

def stand_still_when_zero_command(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    return torch.sum(torch.abs(diff_angle), dim=1) * command



def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_clearance(
    env: ManagerBasedRLEnv,
    asset_feet_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_feet_height: float = 0.35
) -> torch.Tensor:

    asset_feet: Articulation = env.scene[asset_feet_cfg.name]
    asset_base: Articulation = env.scene[asset_base_cfg.name]

    feet_positions = asset_feet.data.body_pos_w[:, asset_feet_cfg.body_ids, :] # (num_envs, num_feet, 3)
    feet_vels = asset_feet.data.body_lin_vel_w[:, asset_feet_cfg.body_ids, :] # (num_envs, num_feet, 3)

    base_rotation = asset_base.data.root_link_quat_w[:, :] # (num_envs, 4)
    base_positions = asset_base.data.root_link_pos_w[:, :] # (num_envs, 3)
    base_vels = asset_base.data.root_link_lin_vel_w[:, :] # (num_envs, 3)

    num_envs = feet_positions.shape[0]
    num_feet = feet_positions.shape[1]
    cur_footpos_translated = feet_positions - base_positions.unsqueeze(1)
    footpos_in_body_frame = torch.zeros(num_envs, num_feet, 3, device='cuda')
    cur_footvel_translated = feet_vels - base_vels.unsqueeze(1)
    footvel_in_body_frame = torch.zeros(num_envs, num_feet, 3, device='cuda')
    for i in range(num_feet):
        footpos_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(base_rotation, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(base_rotation, cur_footvel_translated[:, i, :])

    height_error = torch.square(footpos_in_body_frame[:, :, 2] - target_feet_height).view(num_envs, -1)
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(num_envs, -1)

    return torch.sum(height_error * foot_leteral_vel, dim=1)


class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty




class GaitRewardQuad(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.asset_base_cfg = cfg.params["asset_base_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.asset_base: Articulation = env.scene[self.asset_base_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        # self.height_scale = float(cfg.params["tracking_contacts_shaped_height"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        # self.height_target = cfg.params["gait_height_target"]
        # self.height_sigma = cfg.params["gait_height_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRtLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        # tracking_contacts_shaped_height,
        gait_force_sigma,
        gait_vel_sigma,
        # gait_height_target,
        # gait_height_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
        asset_base_cfg,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1) # (num_envs, num_feet)
        # print(foot_forces.shape, desired_contact_states.shape)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        # body_lin_vel_w (num_envs, num_feet, 3)
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, 0:2], dim=-1) # (num_envs, num_feet)
        # print("foot_velocities {}".format(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, 0:2].shape))
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)
        

        # feet_positions = self.asset.data.body_pos_w[:, asset_cfg.body_ids, :] # (num_envs, num_feet, 3)

        # base_rotation = self.asset_base.data.root_link_quat_w[:, :] # (num_envs, 4)
        # base_positions = self.asset_base.data.root_link_pos_w[:, :] # (num_envs, 3)
        # num_envs = feet_positions.shape[0]
        # num_feet = feet_positions.shape[1]
        # cur_footpos_translated = feet_positions - base_positions.unsqueeze(1)
        # footpos_in_body_frame = torch.zeros(num_envs, num_feet, 3, device='cuda')
        # for i in range(num_feet):
        #     footpos_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(base_rotation, cur_footpos_translated[:, i, :]) 

        # print(footpos_in_body_frame[:, :, 2][0])
        
        # foots_error = footpos_in_body_frame[:, :, 2] - self.height_target
        # feet_height_reward = self._compute_foot_height_reward(foots_error, desired_contact_states)
        # Combine rewards
        total_reward = force_reward + velocity_reward
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        durations = torch.cat(
            [
                gait_params[:, 1].view(self.num_envs, 1),
                gait_params[:, 1].view(self.num_envs, 1),
                gait_params[:, 1].view(self.num_envs, 1),
                gait_params[:, 1].view(self.num_envs, 1),
            ],
            dim=1,
        )
        offsets2 = gait_params[:, 2]
        offsets3 = gait_params[:, 3]
        offsets4 = gait_params[:, 4]

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets2 >= 0) & (offsets2 <= 1)), "Offsets2 must be between 0 and 1"
        assert torch.all((offsets3 >= 0) & (offsets3 <= 1)), "Offsets3 must be between 0 and 1"
        assert torch.all((offsets4 >= 0) & (offsets4 <= 1)), "Offsets4 must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [
                    gait_indices.view(self.num_envs, 1),
                    (gait_indices + offsets2 + 1).view(self.num_envs, 1),
                    (gait_indices + offsets3 + 1).view(self.num_envs, 1),
                    (gait_indices + offsets4 + 1).view(self.num_envs, 1)
                ],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                # print("i: {} forces[:, i] ** 2: {}".format(i, forces[:, i] ** 2))
                # 摆动相位中的不需要的接触
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                # print("i: {} velocities[:, i] ** 2: {}".format(i, velocities[:, i] ** 2))
                # 惩罚支撑相的速度
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale
    
    def _compute_foot_height_reward(self, foots_error: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute foot height-based reward component."""
        reward = torch.zeros_like(foots_error[:, 0])
        if self.height_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(foots_error.shape[1]):
            
                # print("i: {} forces[:, i] ** 2: {}".format(i, forces[:, i] ** 2))
                # 摆动相位中的不需要的接触
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-foots_error[:, i] ** 2 / self.height_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(foots_error.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-foots_error[:, i] ** 2 / self.height_sigma)
        return (reward / foots_error.shape[1]) * self.height_scale






class GaitRewardQuad_NoCommand(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt
        self.basic_command_name = cfg.params["basic_command_name"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
        basic_command_name,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1) # (num_envs, num_feet)
        # print(foot_forces.shape, desired_contact_states.shape)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        # body_lin_vel_w (num_envs, num_feet, 3)
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, 0:2], dim=-1) # (num_envs, num_feet)
        # print("foot_velocities {}".format(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, 0:2].shape))
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        command = torch.norm(env.command_manager.get_command(self.basic_command_name)[:, :2], dim=1) < 0.1
        
        # Combine rewards
        total_reward = (force_reward + velocity_reward) * command
        # print("command: {} total_reward: {} ".format(self.command[0], (force_reward*velocity_reward)[0]))
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        durations = torch.cat(
            [
                gait_params[:, 1].view(self.num_envs, 1),
                gait_params[:, 1].view(self.num_envs, 1),
                gait_params[:, 1].view(self.num_envs, 1),
                gait_params[:, 1].view(self.num_envs, 1),
            ],
            dim=1,
        )
        offsets2 = gait_params[:, 2]
        offsets3 = gait_params[:, 3]
        offsets4 = gait_params[:, 4]

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets2 >= 0) & (offsets2 <= 1)), "Offsets2 must be between 0 and 1"
        assert torch.all((offsets3 >= 0) & (offsets3 <= 1)), "Offsets3 must be between 0 and 1"
        assert torch.all((offsets4 >= 0) & (offsets4 <= 1)), "Offsets4 must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [
                    gait_indices.view(self.num_envs, 1),
                    (gait_indices + offsets2 + 1).view(self.num_envs, 1),
                    (gait_indices + offsets3 + 1).view(self.num_envs, 1),
                    (gait_indices + offsets4 + 1).view(self.num_envs, 1)
                ],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                # print("i: {} forces[:, i] ** 2: {}".format(i, forces[:, i] ** 2))
                # 摆动相位中的不需要的接触
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                # print("i: {} velocities[:, i] ** 2: {}".format(i, velocities[:, i] ** 2))
                # 惩罚支撑相的速度
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale
    
