import logging
import os
import time
import numpy as np
import onnxruntime as ort
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import FalconPolicyCfg
from robojudo.utils.util_func import command_remap, get_gravity_orientation
logger = logging.getLogger(__name__)


@policy_registry.register
class FalconPolicy(Policy):
    cfg_policy: FalconPolicyCfg

    def __init__(self, cfg_policy: FalconPolicyCfg, device):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        logger.debug(f"Loading policy '{cfg_policy.policy_name}' from {cfg_policy.policy_file}")
        self.session = ort.InferenceSession(cfg_policy.policy_file)

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        super().__init__(cfg_policy=cfg_policy, device=device)

        self.obs_scales = self.cfg_policy.obs_scales
        self.command_ranges=self.cfg_policy.command_ranges
        self.ref_upper_dof_pos=np.zeros(14)
        self.use_history = cfg_policy.USE_HISTORY

        self.reset()

    def reset(self):
        self.last_action = np.zeros(self.num_actions)
        if self.use_history:
            self.history_obs_dims = self.cfg_policy.history_obs_dims
            default_history = [np.zeros(dim, dtype=np.float32) for dim in self.history_obs_dims.values()]
            self._init_history(default_history)

    def post_step_callback(self, commands=None):
        # self.timestep += 1
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {
            "actor_obs": np.expand_dims(obs, axis=0).astype(np.float32),
        }

        ort_outputs = self.session.run(
            ["action"],
            ort_inputs,
        )
        actions: np.ndarray = np.asarray(ort_outputs[0]).squeeze()

        processed_actions = actions
        if self.action_clip is not None:
            processed_actions = np.clip(processed_actions, -self.action_clip, self.action_clip)

        self.last_action = actions.copy()  # TODO: check all policies after process
        processed_actions = processed_actions * self.action_scale
        return processed_actions
    def _get_commands(self, ctrl_data):
        command_lin_vel = np.zeros(2)
        command_ang_vel = np.zeros(1)
        command_stand = np.zeros(1)
        command_waist_dofs=np.zeros(3)
        command_base_height=np.zeros(1)
        for key in ctrl_data.keys():
            if key in ["KeyboardCtrl"]:
                keys = ctrl_data[key]["keyboard_event"]
                for event in keys:
                    if event["type"] == "keyboard":
                        match event["name"]:
                            case "w":
                                command_lin_vel[0] +=0.1
                                command_lin_vel[0] = np.clip(command_lin_vel[0], self.command_ranges.lin_vel_x[0], self.command_ranges.lin_vel_x[1])
                            case "s":
                                command_lin_vel[0] -=0.1
                                command_lin_vel[0] = np.clip(command_lin_vel[0], self.command_ranges.lin_vel_x[0], self.command_ranges.lin_vel_x[1])
                            case "a":
                                command_lin_vel[1] +=0.1
                                command_lin_vel[1] = np.clip(command_lin_vel[1], self.command_ranges.lin_vel_y[0], self.command_ranges.lin_vel_y[1])

                            case "d":
                                command_lin_vel[1] -=0.1
                                command_lin_vel[1] = np.clip(command_lin_vel[1], self.command_ranges.lin_vel_y[0], self.command_ranges.lin_vel_y[1])
                            case "e":
                                command_ang_vel[0] +=0.1
                                command_ang_vel[0] = np.clip(command_ang_vel[0], self.command_ranges.ang_vel_yaw[0], self.command_ranges.ang_vel_yaw[1])

                            case "q":
                                command_ang_vel[0] -=0.1
                                command_ang_vel[0] = np.clip(command_ang_vel[0], self.command_ranges.ang_vel_yaw[0], self.command_ranges.ang_vel_yaw[1])

                break
        return command_lin_vel,command_ang_vel,command_stand,command_waist_dofs,command_base_height
    def _get_obs_history(self):
        # for i, items in enumerate(zip(*self.history_buf, strict=True)):
        #     for j, item in enumerate(items):
        #         print(f"items[{i}][{j}] shape: {np.array(item).shape}")
        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        return np.concatenate(history_list, axis=0)
    def get_observation(self, env_data, ctrl_data):
        command_lin_vel,command_ang_vel,command_stand,command_waist_dofs,command_base_height = self._get_commands(ctrl_data)
        if self.use_history:
            history = self._get_obs_history()
            history *= self.obs_scales.history
        else:
            history = []
        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        obs = np.concatenate(
            [
                env_data.base_ang_vel * self.obs_scales.base_ang_vel,
                gravity_orientation,
                command_lin_vel * self.obs_scales.command_lin_vel,
                command_ang_vel * self.obs_scales.command_ang_vel,
                command_stand * self.obs_scales.command_stand,
                command_waist_dofs * self.obs_scales.command_waist_dofs,
                (command_base_height+self.cfg_policy.command_base_height_default)* self.obs_scales.command_base_height,
                self.ref_upper_dof_pos* self.obs_scales.ref_upper_dof_pos,
                env_data.dof_pos - self.default_dof_pos,
                env_data.dof_vel * self.obs_scales.dof_vel,
                self.last_action,
                history,
            ]
        )
        obs_a = [
                env_data.base_ang_vel * self.obs_scales.base_ang_vel,
                gravity_orientation,
                command_lin_vel * self.obs_scales.command_lin_vel,
                command_ang_vel * self.obs_scales.command_ang_vel,
                command_stand * self.obs_scales.command_stand,
                command_waist_dofs * self.obs_scales.command_waist_dofs,
                (command_base_height+self.cfg_policy.command_base_height_default)* self.obs_scales.command_base_height,
                self.ref_upper_dof_pos* self.obs_scales.ref_upper_dof_pos,
                env_data.dof_pos - self.default_dof_pos,
                env_data.dof_vel * self.obs_scales.dof_vel,
                self.last_action,
        ]
        self.history_buf.appendleft(obs_a)
        
        extras = {
            "command_lin_vel": command_lin_vel,
            "command_ang_vel": command_ang_vel,
            "command_stand": command_stand,
            "command_waist_dofs": command_waist_dofs,
            "command_base_height": command_base_height,
        }
        return obs, extras