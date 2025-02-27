import gymnasium as gym

from ext_template.tasks.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg import UnitreeGo1RoughPPORunnerCfg
from ext_template.tasks.locomotion.velocity.config.mi4.agents.rsl_rl_ppo_cfg import MI4RoughPPORunnerCfg
from . import go1_him_env_cfg
from . import mi4_him_env_cfg


"""
Create PPO runners for RSL-RL
"""


go1_blind_rough_runner_cfg_v0 = UnitreeGo1RoughPPORunnerCfg()
go1_blind_rough_runner_cfg_v0.experiment_name = "unitree_go1_rough_v0"
go1_blind_rough_runner_cfg_v0.run_name = "v0"

go1_blind_rough_runner_cfg_v1 = UnitreeGo1RoughPPORunnerCfg()
go1_blind_rough_runner_cfg_v1.experiment_name = "unitree_go1_rough_v1"
go1_blind_rough_runner_cfg_v1.run_name = "v0"



mi4_blind_rough_runner_cfg_v0 = MI4RoughPPORunnerCfg()
mi4_blind_rough_runner_cfg_v0.experiment_name = "mi4_rough_v0"
mi4_blind_rough_runner_cfg_v0.run_name = "v0"


"""
Register Gym environments
"""



# ------------------------------------------------- #
# UNITREE GO1 Blind Rough Environment HIM  v1
# ------------------------------------------------- #
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-HIM-v0",
    entry_point="ext_template.envs:HIMManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_him_env_cfg.UnitreeGo1RoughEnvCfg_v0,
        "rsl_rl_cfg_entry_point": go1_blind_rough_runner_cfg_v0,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-Play-HIM-v0",
    entry_point="ext_template.envs:HIMManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_him_env_cfg.UnitreeGo1RoughEnvCfg_PLAY_v0,
        "rsl_rl_cfg_entry_point": go1_blind_rough_runner_cfg_v0,
    },
)

# ------------------------------------------------- #
# UNITREE GO1 Blind Rough Environment HIM with Gait v1
# ------------------------------------------------- #
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-HIM-v1",
    entry_point="ext_template.envs:HIMManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_him_env_cfg.UnitreeGo1RoughEnvCfg_v1,
        "rsl_rl_cfg_entry_point": go1_blind_rough_runner_cfg_v1,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-Play-HIM-v1",
    entry_point="ext_template.envs:HIMManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_him_env_cfg.UnitreeGo1RoughEnvCfg_PLAY_v1,
        "rsl_rl_cfg_entry_point": go1_blind_rough_runner_cfg_v1,
    },
)

# ------------------------------------------------- #
# MI4 Blind Rough Environment HIM with Gait v0
# ------------------------------------------------- #

gym.register(
    id="Isaac-Velocity-Rough-MI4-HIM-v0",
    entry_point="ext_template.envs:HIMManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": mi4_him_env_cfg.MI4RoughEnvCfg_v0,
        "rsl_rl_cfg_entry_point": mi4_blind_rough_runner_cfg_v0,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-MI4-Play-HIM-v0",
    entry_point="ext_template.envs:HIMManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": mi4_him_env_cfg.MI4RoughEnvCfg_PLAY_v0,
        "rsl_rl_cfg_entry_point": mi4_blind_rough_runner_cfg_v0,
    },
)