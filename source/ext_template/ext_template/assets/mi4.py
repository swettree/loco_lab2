import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os


current_dir = os.path.dirname(__file__)
MI4_USD_PATH = os.path.join(current_dir, "./MI/mi4/USD/mi4.usd")

MI4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MI4_USD_PATH,
        
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=360,  # taken from spec sheet
            velocity_limit=12,  # taken from spec sheet
            saturation_effort=360,  # same as effort limit
            stiffness={
                ".*_hip_joint": 52.0,
                ".*_thigh_joint": 103.0,
                ".*_calf_joint": 97.0,

            },
            damping={
                ".*_hip_joint": 3.6,
                ".*_thigh_joint": 4.6,
                ".*_calf_joint": 4.2,
            },
            friction=0.0,
        ),
    },
)