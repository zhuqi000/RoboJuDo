from robojudo.policy.policy_cfgs import FalconPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1_29FalconDoF(DoFConfig):
    """robot_dof as g1_29dof"""

    joint_names: list[str] = [
        *[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ],
        *[
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        *["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        *[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        *[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    ]

    default_pos: list[float] | None = [
        *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],#right fit
        *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],#right fit
        *[0.0, 0.0, 0.0],#right fit
        *[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],#right fit
        *[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],#right fit
    ]

    stiffness: list[float] | None = [
        *[100, 100, 100, 200, 20, 20],
        *[100, 100, 100, 200, 20, 20],
        *[300, 300, 300],
        *[90, 60, 20, 60, 4, 4, 4],
        *[90, 60, 20, 60, 4, 4, 4],
    ]

    damping: list[float] | None = [
        *[2.5, 2.5, 2.5, 5, 0.2, 0.1],
        *[2.5, 2.5, 2.5, 5, 0.2, 0.1],
        *[5.0, 5.0, 5.0],
        *[2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2],
        *[2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2],
    ]

    position_limits: list[list[float]] | None = [
        *[
            [-2.5307, 2.8798],
            [-0.5236, 2.9671],
            [-2.7576, 2.7576],
            [-0.087267, 2.8798],
            [-0.87267, 0.5236],
            [-0.2618, 0.2618],
        ],
        *[
            [-2.5307, 2.8798],
            [-2.9671, 0.5236],
            [-2.7576, 2.7576],
            [-0.087267, 2.8798],
            [-0.87267, 0.5236],
            [-0.2618, 0.2618],
        ],
        *[[-2.618, 2.618], [-0.52, 0.52], [-0.52, 0.52]],
        *[
            [-3.0892, 2.6704],
            [-1.5882, 2.2515],
            [-2.618, 2.618],
            [-1.0472, 2.0944],
            [-1.972222054, 1.972222054],
            [-1.614429558, 1.614429558],
            [-1.614429558, 1.614429558],
        ],
        *[
            [-3.0892, 2.6704],
            [-2.2515, 1.5882],
            [-2.618, 2.618],
            [-1.0472, 2.0944],
            [-1.972222054, 1.972222054],
            [-1.614429558, 1.614429558],
            [-1.614429558, 1.614429558],
        ],
    ]

    torque_limits: list[float] | None = [
        *[88.0, 88.0, 88.0, 139.0, 50.0, 50.0],
        *[88.0, 88.0, 88.0, 139.0, 50.0, 50.0],
        *[88.0, 50.0, 50.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0],
    ]

# TODO:my
class G1FalconPolicyCfg(FalconPolicyCfg):
    robot: str = "g1"

    # ======= MOTION CONFIGURATION =======
    policy_name: str = "g1_29dof"

    # ======= POLICY DOF CONFIGURATION =======

    obs_dof: DoFConfig = G1_29FalconDoF()
    action_dof: DoFConfig = obs_dof

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: FalconPolicyCfg.ObsScalesCfg = FalconPolicyCfg.ObsScalesCfg(
        base_ang_vel=0.25,
        projected_gravity=1.0,
        command_lin_vel=1.0,
        command_ang_vel=1.0,
        command_stand=1.0,
        command_waist_dofs=1.0,
        command_base_height=2.0,# only apply the base height if standing
        ref_upper_dof_pos=1.0,
        dof_pos=1.0,
        dof_vel=0.05,
        history=1.0,
        actions=1.0,
    )

    history_length: int = 4
    history_obs_dims: dict[str, int] = {  # from obs_mimic_dims, SORTED by key!!!
        "actions":action_dof.num_dofs,           
        "base_ang_vel": 3,
        "command_ang_vel": 1,
        "command_base_height": 1,  
        "command_lin_vel": 2,
        "command_stand": 1,
        "command_waist_dofs": 3,  # only apply the base height if standing
        "dof_pos":obs_dof.num_dofs,
        "dof_vel":obs_dof.num_dofs,         
        "projected_gravity": 3,
        "ref_upper_dof_pos":14, 
    }

    USE_HISTORY: bool = True
    GAIT_PERIOD: float=1.0
    command_base_height_default: float=0.75
    command_ranges:FalconPolicyCfg.locomotion_command_ranges=FalconPolicyCfg.locomotion_command_ranges(
        lin_vel_x=[-1.0, 1.0],
        lin_vel_y=[-1.0, 1.0],
        ang_vel_yaw=[-1.0, 1.0],
        heading=[-3.14, 3.14],
        base_height=[-0.25, 0.0],
    )
 
