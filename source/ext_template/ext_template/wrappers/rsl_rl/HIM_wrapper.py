# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure a :class:`ManagerBasedRLEnv` or :class:`DirectRlEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""


import gymnasium as gym
import torch

from rsl_rl.env import VecEnv

from ext_template.envs import  HIMManagerBasedRLEnv
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

class HIMRslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for RSL-RL library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "critic" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv | HIMManagerBasedRLEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv) and not isinstance(env.unwrapped, HIMManagerBasedRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        # 定义 policy 和 critic 的观测组件顺序
        self.policy_keys = [
            "velocity_commands",   # 3 维
            "gait_phase",          # 2 维
            "gait_command",        # 5 维
            "base_ang_vel",        # 3 维
            "projected_gravity",   # 3 维
            "joint_pos",           # 12 维
            "joint_vel",           # 12 维
            "actions"              # 12 维
        ]
        
        self.critic_keys = [
            "velocity_commands",   # 3 维
            "gait_phase",          # 2 维
            "gait_command",        # 5 维
            "base_ang_vel",        # 3 维
            "projected_gravity",   # 3 维
            "joint_pos",           # 12 维
            "joint_vel",           # 12 维
            "actions",             # 12 维
            "base_lin_vel",        # 3 维
            "height_scan"          # 187 维
        ]
        # 获取一次观测以确定 num_obs

        self.policy_history_length = 5
        self.critic_history_length = 1
        sample_policy_obs, sample_critic_obs = self.get_observations()
        self.policy_num_obs = sample_policy_obs.shape[1]  # 225 # 
        self.critic_num_obs = sample_critic_obs.shape[1]  # 235


        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            # self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
            self.num_obs = self.policy_num_obs
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            # self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
            self.num_privileged_obs = self.critic_num_obs
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """
    def process_observation(self, obs_dict, keys, history_length, flip_time=True):
        """
        处理观测数据，将指定的观测组件按顺序拼接，并展平成指定的形状。
        
        参数:
            obs_dict (dict): 观测字典，包含各个观测组件。
            keys (list): 需要处理的观测组件名称列表。
            history_length (int): 历史长度。
            flip_time (bool): 是否翻转时间维度以确保左新右旧。
        
        返回:
            torch.Tensor: 处理后的观测张量，形状为 [num_envs, n]，其中 n = history_length * sum(feature_dims)
        """
    
        if not obs_dict:
            return torch.tensor([])

        obs_list = []
        num_envs = None    
        
        for key in keys:
            if key not in obs_dict:
                raise KeyError(f"Observation key '{key}' not found in the observation dictionary.")
            obs = obs_dict[key]

            if num_envs is None:
                num_envs = obs.size(0)
            elif obs.size(0) != num_envs:
                raise ValueError(f"All observation components must have the same num_envs.")
                
            if obs.size(1) != history_length:
                raise ValueError(f"Expected history_length={history_length} for '{key}', but got {obs.size(1)}.")

            if flip_time:
                obs = torch.flip(obs, dims=[1])   # 调整时间顺序（确保左新右旧）
            obs_list.append(obs)    

        # 将每个key的obs按时间维度拼接
        time_step_obs_list = [torch.cat([obs[:, t, :] for obs in obs_list], dim=1) for t in range(history_length)]
        
        combined_obs = torch.cat(time_step_obs_list, dim=1) # 在特征维度上拼接所有时间步的obs  
        return combined_obs




    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the current observations of the environment as policy and critic observations.
        
        Policy Observations:
            - History Length: 5
            - Keys: ["velocity_commands", "base_ang_vel", "projected_gravity", 
                    "joint_pos", "joint_vel", "actions"]
        
        Critic Observations:
            - History Length: 1
            - Keys: ["velocity_commands", "base_ang_vel", "projected_gravity", 
                    "joint_pos", "joint_vel", "actions", "base_lin_vel", "height_scan"]
        
        Both observations ensure time order is from left (newest) to right (oldest).
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (policy_obs, critic_obs)
        """
        # 获取环境的原始观测
        if hasattr(self.unwrapped, "observation_manager"):
            full_obs_dict = self.unwrapped.observation_manager.compute()
        else:
            full_obs_dict = self.unwrapped._get_observations()
        
        # 分别获取 policy 和 critic 的观测字典
        policy_obs_dict = full_obs_dict.get("policy", {})
        critic_obs_dict = full_obs_dict.get("critic", {})
        
        if not policy_obs_dict:
            raise KeyError("No 'policy' observations found in the observation dictionary.")
        if not critic_obs_dict:
            raise KeyError("No 'critic' observations found in the observation dictionary.")
        
        # 处理 policy 观测
        policy_obs = self.process_observation(
            obs_dict=policy_obs_dict,
            keys=self.policy_keys,
            history_length=self.policy_history_length,
            flip_time=True  # 确保时间顺序为左新右旧
        )
        # 处理 critic 观测
        critic_obs = self.process_observation(
            obs_dict=critic_obs_dict,
            keys=self.critic_keys,
            history_length=self.critic_history_length,
            flip_time=True  # 即使 history_length=1，保持一致性
        )

        return policy_obs, critic_obs

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    # ding
    @property
    def num_one_step_obs(self) -> int:
        """Returns the number of one-step observations."""
        # 确保history_length不为零，避免除零错误
        # history_length = self.unwrapped.observation_manager.actor.history_length
        # if history_length == 0:
        #     return self.num_obs
        
        # 使用整数除法
        
        return self.policy_num_obs //self.policy_history_length

    @property
    def num_one_step_privileged_obs(self) -> int:
        """Returns the number of one-step privileged observations."""
        # history_length = self.unwrapped.observation_manager.critic.history_length
        # if not history_length:
        #     return self.num_privileged_obs
 
        return  self.critic_num_obs // self.critic_history_length



    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # return observations
        return obs_dict["policy"], obs_dict["critic"]

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_dict, rew, terminated, truncated, extras  = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
                # 分别获取 policy 和 critic 的观测字典
        policy_obs_dict = obs_dict.get("policy", {})
        critic_obs_dict = obs_dict.get("critic", {})
        critic_obs_before_reset_dict = extras.get("critic_before_reset", {})
        # print("vel0", policy_obs_dict["projected_gravity"][0])
        obs = self.process_observation(
            obs_dict=policy_obs_dict,
            keys=self.policy_keys,
            history_length=self.policy_history_length,
            flip_time=True  # 确保时间顺序为左新右旧
        )

        critic_obs = self.process_observation(
            obs_dict=critic_obs_dict,
            keys=self.critic_keys,
            history_length=self.critic_history_length,
            flip_time=True  # 即使 history_length=1，保持一致性
        )

        critic_obs_before_reset = self.process_observation(
            obs_dict=critic_obs_before_reset_dict,
            keys=self.critic_keys,
            history_length=self.critic_history_length,
            flip_time=True  # 即使 history_length=1，保持一致性
        )
        # if "critic" in obs_dict:
        #     privileged_obs = obs_dict["critic"]
        # else:
        #     privileged_obs = obs
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # print("vel1", obs[0])
        extras["critic_buffer_before_reset"] = critic_obs_before_reset
        # return the step information
        return obs, critic_obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()
