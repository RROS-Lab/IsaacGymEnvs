# Copyright (c) 2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""IndustReal: base class.

Inherits Factory base class and Factory abstract base class. Inherited by IndustReal environment classes. Not directly executed.

Configuration defined in IndustRealBase.yaml. Asset info defined in industreal_asset_info_franka_table.yaml.
"""


import hydra
import math
import os
import torch

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_kuka_base import FactoryKukaBase
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_schema_class_base import FactoryABCBase
from isaacgymenvs.tasks.factory.factory_schema_config_base import (
    FactorySchemaConfigBase,
)


class IndustRealKukaBase(FactoryKukaBase, FactoryABCBase):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        """Initialize instance variables. Initialize VecTask superclass."""

        self.cfg = cfg
        self.cfg["headless"] = headless

        self._get_base_yaml_params()

        if self.cfg_base.mode.export_scene:
            sim_device = "cpu"

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )  # create_sim() is called here

    def _get_base_yaml_params(self):
        # TODO(dhanush): Refactor
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_base", node=FactorySchemaConfigBase)

        config_path = (
            "task/IndustRealBase.yaml"  # relative to Gym's Hydra search path (cfg dir)
        )
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base["task"]  # strip superfluous nesting

        asset_info_path = "../../assets/industreal/yaml/industreal_asset_info_franka_table.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_franka_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_franka_table = self.asset_info_franka_table[""][""][""][""][""][
            ""
        ]["assets"]["industreal"][
            "yaml"
        ]  # strip superfluous nesting

    def import_kuka_assets(self):
        # TODO(dhanush): Validate
        """Set KUKA and table asset options. Import assets."""

        urdf_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "assets", "industreal_kuka", "urdf"
        )

        kuka_file = "kuka.urdf"

        kuka_options = gymapi.AssetOptions()
        kuka_options.flip_visual_attachments = True
        kuka_options.fix_base_link = True
        kuka_options.collapse_fixed_joints = False
        kuka_options.thickness = 0.0  # default = 0.02
        kuka_options.density = 1000.0  # default = 1000.0
        kuka_options.armature = 0.01  # default = 0.0
        kuka_options.use_physx_armature = True
        if self.cfg_base.sim.add_damping:
            kuka_options.linear_damping = (
                1.0  # default = 0.0; increased to improve stability
            )
            kuka_options.max_linear_velocity = (
                1.0  # default = 1000.0; reduced to prevent CUDA errors
            )
            kuka_options.angular_damping = (
                5.0  # default = 0.5; increased to improve stability
            )
            kuka_options.max_angular_velocity = (
                2 * math.pi
            )  # default = 64.0; reduced to prevent CUDA errors
        else:
            kuka_options.linear_damping = 0.0  # default = 0.0
            kuka_options.max_linear_velocity = 1.0  # default = 1000.0
            kuka_options.angular_damping = 0.5  # default = 0.5
            kuka_options.max_angular_velocity = 2 * math.pi  # default = 64.0
        kuka_options.disable_gravity = True
        kuka_options.enable_gyroscopic_forces = True
        kuka_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        kuka_options.use_mesh_materials = True
        if self.cfg_base.mode.export_scene:
            kuka_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        table_options = gymapi.AssetOptions()
        table_options.flip_visual_attachments = False  # default = False
        table_options.fix_base_link = True
        table_options.thickness = 0.0  # default = 0.02
        table_options.density = 1000.0  # default = 1000.0
        table_options.armature = 0.0  # default = 0.0
        table_options.use_physx_armature = True
        table_options.linear_damping = 0.0  # default = 0.0
        table_options.max_linear_velocity = 1000.0  # default = 1000.0
        table_options.angular_damping = 0.0  # default = 0.5
        table_options.max_angular_velocity = 64.0  # default = 64.0
        table_options.disable_gravity = False
        table_options.enable_gyroscopic_forces = True
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            table_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        kuka_asset = self.gym.load_asset(
            self.sim, urdf_root, kuka_file, kuka_options
        )
        table_asset = self.gym.create_box(
            self.sim,
            self.asset_info_franka_table.table_depth,
            self.asset_info_franka_table.table_width,
            self.cfg_base.env.table_height,
            table_options,
        )

        return kuka_asset, table_asset

    def acquire_base_tensors(self):
        # TODO(dhanush): Refactor
        """Acquire and wrap tensors. Create views."""

        _root_state = self.gym.acquire_actor_root_state_tensor(
            self.sim
        )  # shape = (num_envs * num_actors, 13)
        _body_state = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )  # shape = (num_envs * num_bodies, 13)
        _dof_state = self.gym.acquire_dof_state_tensor(
            self.sim
        )  # shape = (num_envs * num_dofs, 2)
        _dof_force = self.gym.acquire_dof_force_tensor(
            self.sim
        )  # shape = (num_envs * num_dofs, 1)
        _contact_force = self.gym.acquire_net_contact_force_tensor(
            self.sim
        )  # shape = (num_envs * num_bodies, 3)
        _jacobian = self.gym.acquire_jacobian_tensor(
            self.sim, "kuka"  # NOTE(dhanush): _create_actors -> from env_pegs
        )  # shape = (num envs, num_bodies, 6, num_dofs)
        _mass_matrix = self.gym.acquire_mass_matrix_tensor(
            self.sim, "kuka"  # NOTE(dhanush): _create_actors -> from env_pegs
        )  # shape = (num_envs, num_dofs, num_dofs)

        self.root_state = gymtorch.wrap_tensor(_root_state)
        self.body_state = gymtorch.wrap_tensor(_body_state)
        self.dof_state = gymtorch.wrap_tensor(_dof_state)
        self.dof_force = gymtorch.wrap_tensor(_dof_force)
        self.contact_force = gymtorch.wrap_tensor(_contact_force)
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        self.mass_matrix = gymtorch.wrap_tensor(_mass_matrix)

        self.root_pos = self.root_state.view(self.num_envs, self.num_actors, 13)[
            ..., 0:3
        ]
        self.root_quat = self.root_state.view(self.num_envs, self.num_actors, 13)[
            ..., 3:7
        ]
        self.root_linvel = self.root_state.view(self.num_envs, self.num_actors, 13)[
            ..., 7:10
        ]
        self.root_angvel = self.root_state.view(self.num_envs, self.num_actors, 13)[
            ..., 10:13
        ]
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[
            ..., 0:3
        ]
        self.body_quat = self.body_state.view(self.num_envs, self.num_bodies, 13)[
            ..., 3:7
        ]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[
            ..., 7:10
        ]
        self.body_angvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[
            ..., 10:13
        ]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_force_view = self.dof_force.view(self.num_envs, self.num_dofs, 1)[
            ..., 0
        ]
        self.contact_force = self.contact_force.view(self.num_envs, self.num_bodies, 3)[
            ..., 0:3
        ]

        # NOTE(dhanush) : KUKA has 8 links -> base_link, {link0, ..., link7}, link_ee
        # TODO(dhanush) : Validate this
        self.arm_dof_pos = self.dof_pos[:, 1:9]
        self.arm_dof_vel = self.dof_vel[:, 1:9]
        self.arm_mass_matrix = self.mass_matrix[
            :, 1:9, 1:9
        ]  # for KUKA arm (not gripper)

        # TODO(dhanush): Figure out these id_envs
        # --------------------------------------- #
        self.robot_base_pos = self.body_pos[:, self.robot_base_body_id_env, 0:3]
        self.robot_base_quat = self.body_quat[:, self.robot_base_body_id_env, 0:4]

        self.hand_pos = self.body_pos[:, self.hand_body_id_env, 0:3]
        self.hand_quat = self.body_quat[:, self.hand_body_id_env, 0:4]
        self.hand_linvel = self.body_linvel[:, self.hand_body_id_env, 0:3]
        self.hand_angvel = self.body_angvel[:, self.hand_body_id_env, 0:3]
        self.hand_jacobian = self.jacobian[
            :, self.hand_body_id_env_actor - 1, 0:6, 0:7
        ]  # minus 1 because base is fixed

        self.left_finger_pos = self.body_pos[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_quat = self.body_quat[:, self.left_finger_body_id_env, 0:4]
        self.left_finger_linvel = self.body_linvel[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_angvel = self.body_angvel[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_jacobian = self.jacobian[
            :, self.left_finger_body_id_env_actor - 1, 0:6, 0:7
        ]  # minus 1 because base is fixed

        self.right_finger_pos = self.body_pos[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_quat = self.body_quat[:, self.right_finger_body_id_env, 0:4]
        self.right_finger_linvel = self.body_linvel[
            :, self.right_finger_body_id_env, 0:3
        ]
        self.right_finger_angvel = self.body_angvel[
            :, self.right_finger_body_id_env, 0:3
        ]
        self.right_finger_jacobian = self.jacobian[
            :, self.right_finger_body_id_env_actor - 1, 0:6, 0:7
        ]  # minus 1 because base is fixed

        self.left_finger_force = self.contact_force[
            :, self.left_finger_body_id_env, 0:3
        ]
        self.right_finger_force = self.contact_force[
            :, self.right_finger_body_id_env, 0:3
        ]

        self.gripper_dof_pos = self.dof_pos[:, 7:9]

        self.fingertip_centered_pos = self.body_pos[
            :, self.fingertip_centered_body_id_env, 0:3
        ]
        self.fingertip_centered_quat = self.body_quat[
            :, self.fingertip_centered_body_id_env, 0:4
        ]
        self.fingertip_centered_linvel = self.body_linvel[
            :, self.fingertip_centered_body_id_env, 0:3
        ]
        self.fingertip_centered_angvel = self.body_angvel[
            :, self.fingertip_centered_body_id_env, 0:3
        ]
        self.fingertip_centered_jacobian = self.jacobian[
            :, self.fingertip_centered_body_id_env_actor - 1, 0:6, 0:7
        ]  # minus 1 because base is fixed

        self.fingertip_midpoint_pos = (
            self.fingertip_centered_pos.detach().clone()
        )  # initial value
        self.fingertip_midpoint_quat = self.fingertip_centered_quat  # always equal
        self.fingertip_midpoint_linvel = (
            self.fingertip_centered_linvel.detach().clone()
        )  # initial value
        # From sum of angular velocities (https://physics.stackexchange.com/questions/547698/understanding-addition-of-angular-velocity),
        # angular velocity of midpoint w.r.t. world is equal to sum of
        # angular velocity of midpoint w.r.t. hand and angular velocity of hand w.r.t. world.
        # Midpoint is in sliding contact (i.e., linear relative motion) with hand; angular velocity of midpoint w.r.t. hand is zero.
        # Thus, angular velocity of midpoint w.r.t. world is equal to angular velocity of hand w.r.t. world.
        self.fingertip_midpoint_angvel = self.fingertip_centered_angvel  # always equal
        self.fingertip_midpoint_jacobian = (
            self.left_finger_jacobian + self.right_finger_jacobian
        ) * 0.5  # approximation

        self.dof_torque = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device
        )
        self.fingertip_contact_wrench = torch.zeros(
            (self.num_envs, 6), device=self.device
        )

        self.ctrl_target_fingertip_centered_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.ctrl_target_fingertip_centered_quat = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.ctrl_target_dof_pos = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device
        )
        self.ctrl_target_gripper_dof_pos = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self.ctrl_target_fingertip_contact_wrench = torch.zeros(
            (self.num_envs, 6), device=self.device
        )

        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        # --------------------------------------- #

    def generate_ctrl_signals(self):
        # TODO(dhanush): Refactor
        """Get Jacobian. Set Franka DOF position targets or DOF torques."""
        # Get desired Jacobian
        if self.cfg_ctrl['jacobian_type'] == 'geometric':
            self.fingertip_midpoint_jacobian_tf = self.fingertip_centered_jacobian
        elif self.cfg_ctrl['jacobian_type'] == 'analytic':
            self.fingertip_midpoint_jacobian_tf = fc.get_analytic_jacobian(
                fingertip_quat=self.fingertip_quat,
                fingertip_jacobian=self.fingertip_centered_jacobian,
                num_envs=self.num_envs,
                device=self.device)
        # Set PD joint pos target or joint torque
        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            self._set_dof_pos_target()
        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            self._set_dof_torque()

    def _set_dof_pos_target(self):
        """Set Franka DOF position target to move fingertips towards target pose."""
        self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl=self.cfg_ctrl,
            arm_dof_pos=self.arm_dof_pos,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            device=self.device)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ctrl_target_dof_pos),
                                                        gymtorch.unwrap_tensor(self.franka_actor_ids_sim),
                                                        len(self.franka_actor_ids_sim))
    def _set_dof_torque(self):
        """Set Franka DOF torque to move fingertips towards target pose."""
        self.dof_torque = fc.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            fingertip_midpoint_linvel=self.fingertip_centered_linvel,
            fingertip_midpoint_angvel=self.fingertip_centered_angvel,
            left_finger_force=self.left_finger_force,
            right_finger_force=self.right_finger_force,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_fingertip_contact_wrench=self.ctrl_target_fingertip_contact_wrench,
            device=self.device)
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_torque),
                                                        gymtorch.unwrap_tensor(self.franka_actor_ids_sim),
                                                        len(self.franka_actor_ids_sim))

    def simulate_and_refresh(self):
        # TODO(dhanush): Refactor
        """Simulate one step, refresh tensors, and render results."""

        self.gym.simulate(self.sim)
        self.refresh_base_tensors()   # TODO
        self.refresh_env_tensors()  # NOTE(dhanush) : just pass IG
        self._refresh_task_tensors()  # TODO
        self.render()

    def enable_gravity(self):
        """Enable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity = gymapi.Vec3(*self.cfg_base.sim.gravity)
        self.gym.set_sim_params(self.sim, sim_params)

    def attach_to_eef(self, sim_steps):
        # TODO(dhanush): IMPLEMENT this! has to replace close_gripper
        raise NotImplementedError
    
    def pose_world_to_robot_base(self, pos, quat):
        # NOTE(dhanush): This should be fine
        """Convert pose from world frame to robot base frame."""

        robot_base_transform_inv = torch_utils.tf_inverse(
            self.robot_base_quat, self.robot_base_pos
        )
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(
            robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos
        )

        return pos_in_robot_base, quat_in_robot_base
    
    def move_gripper_to_target_pose(self, gripper_dof_pos, sim_steps):
        # TODO(dhanush): Refactor
        """Move gripper to control target pose."""

        for _ in range(sim_steps):
            # NOTE: midpoint is calculated based on the midpoint between the actual gripper finger pos, 
            # and centered is calculated with the assumption that the gripper fingers are perfectly mirrored.
            # Here we **intentionally** use *_centered_* pos and quat instead of *_midpoint_*,
            # since the fingertips are exactly mirrored in the real world.
            # TODO(dhanush) : Replace with kuka's stuff
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_centered_pos,
                fingertip_midpoint_quat=self.fingertip_centered_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(  # internally calls generate_ctrl_signals()
                actions=actions,
                ctrl_target_gripper_dof_pos=gripper_dof_pos,
                do_scale=False,
            )

            # Simulate one step
            self.simulate_and_refresh()

        # Stabilize Franka
        self.dof_vel[:, :] = 0.0
        self.dof_torque[:, :] = 0.0
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()

        # Set DOF state
        franka_actor_ids_sim = self.franka_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_torque),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    # --------------------------------------- #
    # NOTE: If I understand correctly, these are not needed for us.
    
    def open_gripper(self, sim_steps):
        # NOTE(dhanush): If I understand correctly, this is not needed for us.
        """Open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self.move_gripper_to_target_pose(gripper_dof_pos=0.1, sim_steps=sim_steps)

    def close_gripper(self, sim_steps):
        # NOTE(dhanush): If I understand correctly, this is not needed for us.
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self.move_gripper_to_target_pose(gripper_dof_pos=0.0, sim_steps=sim_steps)