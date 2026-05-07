from robojudo.policy.policy_cfgs import AmpPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1_29AmpDoF(DoFConfig):
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
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],#right fit
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],#right fit
        *[0.0, 0.0, 0.0],#right fit
        *[0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0],#right fit
        *[0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],#right fit
    ]
    stiffness: list[float] | None = [
        *[99.1, 99.1, 40.2, 99.1, 28.5, 28.5],
        *[99.1, 99.1, 40.2, 99.1, 28.5, 28.5],
        *[40.2, 28.5, 28.5],
        *[14.3, 14.3, 14.3, 14.3, 14.3, 8.6, 8.6],
        *[14.3, 14.3, 14.3, 14.3, 14.3, 8.6, 8.6],
    ]

    damping: list[float] | None = [
        *[6.3, 6.3, 2.6, 6.3, 1.8, 1.8],
        *[6.3, 6.3, 2.6, 6.3, 1.8, 1.8],
        *[2.6, 1.8, 1.8],
        *[0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5],
        *[0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5],
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
    torque_limits: list[float] | None = [
        *[139.0, 139.0, 88.0, 139.0, 50.0, 50.0],
        *[139.0, 139.0, 88.0, 139.0, 50.0, 50.0],
        *[88.0, 50.0, 50.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 10.0, 10.0],
        *[25.0, 25.0, 25.0, 25.0, 25.0, 10.0, 10.0],
    ]
    

class G1AmpPolicyCfg(AmpPolicyCfg):
    robot: str = "g1"

    # ======= MOTION CONFIGURATION =======
    policy_name: str = "Unitree-G1-AMP-Flat_model_20000"

    # ======= POLICY DOF CONFIGURATION =======

    obs_dof: DoFConfig = G1_29AmpDoF()
    action_dof: DoFConfig = obs_dof
    
    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: AmpPolicyCfg.ObsScalesCfg = AmpPolicyCfg.ObsScalesCfg(
        actions=1.0,
        base_ang_vel=1.0,
        command_ang_vel=1.0,
        command_lin_vel=1.0,
        joint_pos=1.0,
        joint_vel=1.0,
        projected_gravity=1.0,
        history=1.0,
    )

    history_length: int = 3
    history_obs_dims: dict[str, int] = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos":obs_dof.num_dofs,
        "dof_vel":obs_dof.num_dofs,         
        "actions":action_dof.num_dofs,           
    }

    USE_HISTORY: bool = True
    command_ranges:AmpPolicyCfg.locomotion_command_ranges=AmpPolicyCfg.locomotion_command_ranges(
        lin_vel_x=[-1.0, 1.0],
        lin_vel_y=[-1.0, 1.0],
        ang_vel_yaw=[-1.0, 1.0],
        heading=[-3.14, 3.14],
        base_height=[-0.25, 0.0],
    )
    action_scales: list[float] = [
        *[0.35, 0.35, 0.55, 0.35, 0.44, 0.44],
        *[0.35, 0.35, 0.55, 0.35, 0.44, 0.44],
        *[0.55, 0.44, 0.44],
        *[0.44, 0.44, 0.44, 0.44, 0.44, 0.29, 0.29],
        *[0.44, 0.44, 0.44, 0.44, 0.44, 0.29, 0.29],
    ]