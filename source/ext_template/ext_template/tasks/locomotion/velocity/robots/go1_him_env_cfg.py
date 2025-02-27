from isaaclab.utils import configclass

from ext_template.tasks.locomotion.velocity.config.go1.rough_env_cfg_v0 import UnitreeGo1EnvCfg_v0

from ext_template.tasks.locomotion.velocity.config.go1.rough_env_cfg_v1 import UnitreeGo1EnvCfg_v1
from ext_template.assets import UNITREE_GO1_local_CFG 



@configclass
class UnitreeGo1RoughEnvCfg_v0(UnitreeGo1EnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_local_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


@configclass
class UnitreeGo1RoughEnvCfg_PLAY_v0(UnitreeGo1RoughEnvCfg_v0):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 8
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.push_robot = None


@configclass
class UnitreeGo1RoughEnvCfg_v1(UnitreeGo1EnvCfg_v1):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_local_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


@configclass
class UnitreeGo1RoughEnvCfg_PLAY_v1(UnitreeGo1RoughEnvCfg_v1):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 8
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.push_robot = None
