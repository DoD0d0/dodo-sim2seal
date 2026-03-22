#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import glob
import io
import os
import time

import numpy as np
import rclpy
import torch
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32, Float32MultiArray

# Observation dimensions per policy type
#   stand/walk: lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3) + jpos(8) + jvel(8) + action(8) = 36
#   jump:       gravity(3) + target_pos_b(3) + time_to_go(1) + jpos(8) + jvel(8) + action(8) + height_scan(160) = 191
OBS_DIM = {'stand': 36, 'walk': 36, 'jump': 191}

# Joint limits from URDF (lower, upper) by joint name
JOINT_LIMITS_BY_NAME = {
    'right_joint_1': (-0.3, 0.3),
    'right_joint_2': (-1.5, 1.5),
    'right_joint_3': (-1.5, 1.5),
    'right_joint_4': (-1.5, 1.5),
    'left_joint_1': (-0.3, 0.3),
    'left_joint_2': (-1.5, 1.5),
    'left_joint_3': (-1.5, 1.5),
    'left_joint_4': (-1.5, 1.5),
}

# Typical Isaac Sim articulation order for this USD
DEFAULT_ISAAC_JOINT_ORDER = [
    'left_joint_1', 'right_joint_1',
    'left_joint_2', 'right_joint_2',
    'left_joint_3', 'right_joint_3',
    'left_joint_4', 'right_joint_4',
]

TRAINING_JOINT_ORDERS = {
    'right_left': [
        'right_joint_1', 'right_joint_2', 'right_joint_3', 'right_joint_4',
        'left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4',
    ],
    'left_right': [
        'left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4',
        'right_joint_1', 'right_joint_2', 'right_joint_3', 'right_joint_4',
    ],
    'isaac': DEFAULT_ISAAC_JOINT_ORDER.copy(),
}
HEIGHT_SCAN_DIM = 160  # grid 1.6x1.0 @ 0.1 resolution = 16x10

# Default action scales from training env.yaml
DEFAULT_ACTION_SCALE = {'stand': 0.5, 'walk': 0.5, 'jump': 0.45}


class DodoPolicyController(Node):

    def __init__(self):
        super().__init__('dodo_policy_controller')

        self.declare_parameter('publish_period_ms', 5)
        self.declare_parameter('policy_path', '')
        self.declare_parameter('policy_type', 'stand')
        self.declare_parameter('action_scale', -1.0)  # -1 = use default for policy_type
        self.declare_parameter('decimation', 4)
        self.declare_parameter('training_joint_order', 'right_left')
        self.declare_parameter('max_policy_action_abs', 2.0)
        self.declare_parameter('max_action_delta', 0.15)
        self.declare_parameter('startup_ramp_sec', 1.0)
        self.declare_parameter('hold_default_pose_sec', 0.5)
        self.declare_parameter('use_cmd_as_lin_vel_when_no_odom', True)
        self.declare_parameter('use_default_cmd_when_no_cmd_vel', True)
        self.declare_parameter('default_cmd_vel_x', 0.25)
        self.declare_parameter('default_cmd_vel_y', 0.0)
        self.declare_parameter('default_cmd_ang_z', 0.0)
        self.declare_parameter('use_first_joint_state_as_default_pos', True)
        self.declare_parameter(
            'default_joint_pos',
            [],
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter('use_steady_time_for_control', True)
        self.declare_parameter('wait_for_cmd_vel', True)
        self.declare_parameter('cmd_vel_to_lin_vel_gain', 0.35)
        self.declare_parameter(
            'joint_signs',
            [1.0] * 8,
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter('imu_axis_remap', 'x,y,z')
        self.declare_parameter('log_first_policy_updates', 12)
        self.set_parameters(
            [rclpy.parameter.Parameter(
                'use_sim_time',
                rclpy.Parameter.Type.BOOL,
                True
            )]
        )

        self._logger = self.get_logger()
        self._policy_type = self.get_parameter('policy_type').value
        self._decimation = self.get_parameter('decimation').value

        # Resolve action scale: use default per policy type if not explicitly set
        scale = self.get_parameter('action_scale').value
        self._action_scale = DEFAULT_ACTION_SCALE.get(self._policy_type, 0.5) if scale < 0 else scale

        self._obs_dim = OBS_DIM.get(self._policy_type, 36)
        self._max_policy_action_abs = float(self.get_parameter('max_policy_action_abs').value)
        self._max_action_delta = float(self.get_parameter('max_action_delta').value)
        self._startup_ramp_sec = float(self.get_parameter('startup_ramp_sec').value)
        self._hold_default_pose_sec = float(self.get_parameter('hold_default_pose_sec').value)
        self._use_cmd_as_lin_vel_when_no_odom = bool(
            self.get_parameter('use_cmd_as_lin_vel_when_no_odom').value
        )
        self._use_default_cmd_when_no_cmd_vel = bool(
            self.get_parameter('use_default_cmd_when_no_cmd_vel').value
        )
        self._use_first_joint_state_as_default_pos = bool(
            self.get_parameter('use_first_joint_state_as_default_pos').value
        )
        self._use_steady_time_for_control = bool(
            self.get_parameter('use_steady_time_for_control').value
        )
        self._wait_for_cmd_vel = bool(self.get_parameter('wait_for_cmd_vel').value)
        self._cmd_vel_to_lin_vel_gain = float(self.get_parameter('cmd_vel_to_lin_vel_gain').value)
        self._imu_axis_remap_spec = str(self.get_parameter('imu_axis_remap').value)
        self._imu_axis_remap = self._parse_axis_remap(self._imu_axis_remap_spec)
        self._log_first_policy_updates = int(self.get_parameter('log_first_policy_updates').value)
        try:
            joint_signs_value = self.get_parameter('joint_signs').value
        except ParameterUninitializedException:
            joint_signs_value = [1.0] * 8

        if isinstance(joint_signs_value, str):
            try:
                parsed_signs = ast.literal_eval(joint_signs_value)
            except (ValueError, SyntaxError):
                parsed_signs = None
            if isinstance(parsed_signs, (list, tuple, np.ndarray)):
                joint_signs = np.array(parsed_signs, dtype=np.float32)
            else:
                joint_signs = np.ones(8, dtype=np.float32)
        else:
            joint_signs = np.array(joint_signs_value, dtype=np.float32)
        if joint_signs.size == 0:
            self._joint_signs = np.ones(8, dtype=np.float32)
        elif joint_signs.size == 8:
            self._joint_signs = joint_signs
        else:
            self._logger.warn(
                "joint_signs must be empty or contain exactly 8 values. Falling back to all +1."
            )
            self._joint_signs = np.ones(8, dtype=np.float32)
        manual_default_pos = np.array(
            self.get_parameter('default_joint_pos').value, dtype=np.float32
        )
        if manual_default_pos.size not in (0, 8):
            self._logger.warn(
                "default_joint_pos must be empty or contain exactly 8 values "
                "(in training joint order). Ignoring invalid input."
            )
            manual_default_pos = np.array([], dtype=np.float32)
        self._default_cmd = np.array(
            [
                float(self.get_parameter('default_cmd_vel_x').value),
                float(self.get_parameter('default_cmd_vel_y').value),
                float(self.get_parameter('default_cmd_ang_z').value),
            ],
            dtype=np.float32,
        )

        # QoS profile for simulation
        sim_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Common subscriptions
        self._cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self._cmd_vel_cb, qos_profile=10)
        self._joint_publisher = self.create_publisher(
            JointState, 'joint_command', qos_profile=sim_qos)
        self._imu_sub = self.create_subscription(
            Imu, 'imu', self._imu_cb, qos_profile=sim_qos)
        self._js_sub = self.create_subscription(
            JointState, 'joint_states', self._joint_states_cb, qos_profile=sim_qos)
        self._odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_cb, qos_profile=sim_qos)

        self._latest_imu: Imu | None = None

        # Jump-specific subscriptions
        self._jump_target = np.zeros(3)  # target_pos in body frame
        self._jump_time_to_go = 0.0
        self._height_scan = np.zeros(HEIGHT_SCAN_DIM)

        if self._policy_type == 'jump':
            self.create_subscription(
                Point, 'jump_target', self._jump_target_cb, qos_profile=10)
            self.create_subscription(
                Float32, 'jump_time', self._jump_time_cb, qos_profile=10)
            self.create_subscription(
                Float32MultiArray, 'height_scan', self._height_scan_cb, qos_profile=sim_qos)

        # Load policy
        self._policy_path = self._resolve_policy_path(
            self.get_parameter('policy_path').value, self._policy_type)
        self.load_policy()

        # State
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._previous_action = np.zeros(8)
        self._policy_counter = 0
        self._policy_update_count = 0
        self._last_tick_time = None  # initialized on first tick
        self._last_tick_time_steady = None
        self._lin_vel_b = np.zeros(3)
        self._dt = 0.0
        self._first_tick = True
        self._warned_zero_cmd = False
        self._warned_no_odom = False
        self._has_odom = False
        self._warned_action_clip = False
        self._warned_cmd_vel_fallback = False
        self._start_time_ros = None
        self._warned_default_cmd = False
        self._has_cmd_vel = False
        self._warned_hold_pose = False
        self._warned_no_imu_fallback = False
        self._seen_imu = False
        self._seen_joint_state = False
        self._warned_waiting_cmd = False

        self.default_pos = np.zeros(8, dtype=np.float32)
        self.action = np.zeros(8, dtype=np.float32)
        self._captured_default_pos = False
        if manual_default_pos.size == 8:
            self.default_pos = manual_default_pos.copy()
            self._captured_default_pos = True

        order_key = self.get_parameter('training_joint_order').value
        self._training_joint_order_key = order_key
        self.joint_names = self._resolve_training_joint_order(order_key)
        self._joint_limits = np.array([JOINT_LIMITS_BY_NAME[n] for n in self.joint_names], dtype=np.float32)

        # Isaac Sim DOF order is finalized from /joint_states at runtime.
        self.isaac_joint_names = DEFAULT_ISAAC_JOINT_ORDER.copy()
        # Mapping: training index → Isaac Sim index
        self._to_isaac_idx = [
            self.isaac_joint_names.index(n) for n in self.joint_names
        ]

        self._logger.info(f"Policy type: {self._policy_type} (obs_dim={self._obs_dim})")
        self._logger.info(f"Policy path: {self._policy_path}")
        self._logger.info(f"Action scale: {self._action_scale}, decimation: {self._decimation}")
        self._logger.info(f"Max policy action abs: {self._max_policy_action_abs}")
        self._logger.info(f"Max action delta: {self._max_action_delta}")
        self._logger.info(f"Startup ramp sec: {self._startup_ramp_sec}")
        self._logger.info(f"Hold default pose sec: {self._hold_default_pose_sec}")
        self._logger.info(
            f"Default cmd fallback enabled: {self._use_default_cmd_when_no_cmd_vel}, "
            f"default_cmd={self._default_cmd.tolist()}"
        )
        if self._captured_default_pos:
            self._logger.info(
                f"Using explicit default_joint_pos: {np.round(self.default_pos, 3).tolist()}"
            )
        else:
            self._logger.info(
                "Default joint offset capture: "
                f"use_first_joint_state_as_default_pos={self._use_first_joint_state_as_default_pos}"
            )
        self._logger.info(
            f"Training joint order ({self._training_joint_order_key}): {self.joint_names}"
        )
        self._logger.info(
            f"use_steady_time_for_control: {self._use_steady_time_for_control}"
        )
        self._logger.info(
            f"wait_for_cmd_vel: {self._wait_for_cmd_vel}, "
            f"cmd_vel_to_lin_vel_gain: {self._cmd_vel_to_lin_vel_gain}"
        )
        self._logger.info(f"joint_signs: {self._joint_signs.tolist()}")
        self._logger.info(f"imu_axis_remap: {self._imu_axis_remap_spec}")
        self._logger.info(f"log_first_policy_updates: {self._log_first_policy_updates}")

    # --- Callbacks ---

    def _imu_cb(self, msg: Imu):
        self._latest_imu = msg
        if not self._seen_imu:
            self._logger.info("First /imu received.")
            self._logger.info(
                f"IMU raw quat (w,x,y,z): [{msg.orientation.w:.4f}, {msg.orientation.x:.4f}, "
                f"{msg.orientation.y:.4f}, {msg.orientation.z:.4f}]"
            )
            self._logger.info(
                f"IMU linear_accel (x,y,z): [{msg.linear_acceleration.x:.3f}, "
                f"{msg.linear_acceleration.y:.3f}, {msg.linear_acceleration.z:.3f}]"
            )
            self._seen_imu = True

    def _joint_states_cb(self, msg: JointState):
        if not self._seen_joint_state:
            self._logger.info("First /joint_states received.")
            self._seen_joint_state = True
        # Do not block control startup waiting for IMU.
        self._tick(msg, self._latest_imu)

    def _cmd_vel_cb(self, msg):
        self._cmd_vel = msg
        self._has_cmd_vel = True

    def _odom_cb(self, msg: Odometry):
        # Prefer body-frame linear velocity from odometry when available.
        self._lin_vel_b = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        self._has_odom = True

    def _jump_target_cb(self, msg):
        self._jump_target = np.array([msg.x, msg.y, msg.z])

    def _jump_time_cb(self, msg):
        self._jump_time_to_go = msg.data

    def _height_scan_cb(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) >= HEIGHT_SCAN_DIM:
            self._height_scan = data[:HEIGHT_SCAN_DIM]

    def _get_cmd_vector(self) -> np.ndarray:
        cmd = np.array(
            [self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z],
            dtype=np.float32,
        )
        if not self._has_cmd_vel and self._use_default_cmd_when_no_cmd_vel:
            cmd = self._default_cmd.copy()
            if not self._warned_default_cmd and self._policy_counter > 50:
                self._logger.warn(
                    "No /cmd_vel received yet; using default_cmd fallback."
                )
                self._warned_default_cmd = True
        return cmd

    # --- Main loop ---

    def _tick(self, joint_state: JointState, imu: Imu | None):
        if self._first_tick:
            self._first_tick = False
            self._refresh_isaac_joint_order(list(joint_state.name))
            if (not self._captured_default_pos) and self._use_first_joint_state_as_default_pos:
                jpos0, _ = self._get_joint_state(joint_state)
                self.default_pos = jpos0.astype(np.float32)
                self._captured_default_pos = True
                self._logger.info(
                    "Captured default_joint_pos from first /joint_states: "
                    f"{np.round(self.default_pos, 3).tolist()}"
                )
            self._logger.info(f"Joint names from /joint_states: {self.isaac_joint_names}")
            self._logger.info(f"Training order (policy):         {self.joint_names}")
            if self.isaac_joint_names != self.joint_names:
                self._logger.warn(
                    "Joint order differs between Isaac Sim and policy order. "
                    "Using name-based remap for /joint_command."
                )

        now_ros = self.get_clock().now().nanoseconds * 1e-9
        now_steady = time.monotonic()
        now = now_steady if self._use_steady_time_for_control else now_ros
        if self._start_time_ros is None:
            self._start_time_ros = now
        if self._last_tick_time is None:
            self._last_tick_time = now
        if self._use_steady_time_for_control:
            if self._last_tick_time_steady is None:
                self._last_tick_time_steady = now_steady
            self._dt = max(0.0, now_steady - self._last_tick_time_steady)
            self._last_tick_time_steady = now_steady
        else:
            if now < self._last_tick_time:
                self._logger.warn(
                    f'{self._get_stamp_prefix()} Time jumped backwards, clamping dt to zero.'
                )
                self._dt = 0.0
            else:
                self._dt = now - self._last_tick_time
        self._last_tick_time = now

        # Compute observation and run policy
        obs = self._compute_observation(joint_state, imu)

        elapsed = now - self._start_time_ros if self._start_time_ros is not None else 0.0
        hold_default_pose = self._hold_default_pose_sec > 0 and elapsed < self._hold_default_pose_sec
        waiting_for_cmd = (
            self._wait_for_cmd_vel
            and self._policy_type in ('stand', 'walk')
            and not self._has_cmd_vel
            and not self._use_default_cmd_when_no_cmd_vel
        )

        if hold_default_pose or waiting_for_cmd:
            self.action = np.zeros(8, dtype=np.float32)
            self._previous_action = self.action.copy()
            if hold_default_pose and not self._warned_hold_pose:
                self._logger.info(
                    f"Holding default pose for {self._hold_default_pose_sec:.2f}s before policy control."
                )
                self._warned_hold_pose = True
            if waiting_for_cmd and not self._warned_waiting_cmd:
                self._logger.warn(
                    "Waiting for first /cmd_vel before enabling policy updates "
                    "(use_default_cmd_when_no_cmd_vel is false)."
                )
                self._warned_waiting_cmd = True
        elif self._policy_counter % self._decimation == 0:
            raw_action = self._compute_action(obs)
            raw_action_max = float(np.max(np.abs(raw_action)))
            if self._max_policy_action_abs > 0:
                target_action = np.clip(
                    raw_action,
                    -self._max_policy_action_abs,
                    self._max_policy_action_abs,
                )
                if (
                    not self._warned_action_clip
                    and raw_action_max > self._max_policy_action_abs
                ):
                    self._logger.warn(
                        f"Policy action exceeded max_policy_action_abs (max_abs={raw_action_max:.3f}) and was clipped. "
                        "This usually indicates observation mismatch."
                    )
                    self._warned_action_clip = True
            else:
                target_action = raw_action

            # Limit per-step action jumps to avoid instant aggressive kicks.
            if self._max_action_delta > 0:
                delta = np.clip(
                    target_action - self.action,
                    -self._max_action_delta,
                    self._max_action_delta,
                )
                self.action = self.action + delta
            else:
                self.action = target_action
            self._previous_action = self.action.copy()
            if self._policy_update_count < self._log_first_policy_updates:
                sim_action = self.action * self._joint_signs
                # Full policy input vector.
                self._logger.info(f"obs:    {np.round(obs, 3).tolist()}")
                # Network output before clip and rate limits.
                self._logger.info(f"raw:    {np.round(raw_action, 3).tolist()} (max_abs={raw_action_max:.3f})")
                # Filtered action in policy joint frame.
                self._logger.info(f"action(policy): {np.round(self.action, 3).tolist()}")
                # Action remapped to simulator joint frame.
                self._logger.info(f"action(sim):    {np.round(sim_action, 3).tolist()}")
            self._policy_update_count += 1
        self._policy_counter += 1

        # Publish joint command in Isaac Sim DOF order
        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.isaac_joint_names
        ramp = 1.0
        if self._startup_ramp_sec > 0 and self._start_time_ros is not None:
            ramp = np.clip((now - self._start_time_ros) / self._startup_ramp_sec, 0.0, 1.0)
        sim_action = self.action * self._joint_signs
        action_pos = np.clip(
            self.default_pos + sim_action * self._action_scale * ramp,
            self._joint_limits[:, 0], self._joint_limits[:, 1]
        )
        reordered = np.zeros(8)
        for train_idx, isaac_idx in enumerate(self._to_isaac_idx):
            reordered[isaac_idx] = action_pos[train_idx]
        self._joint_command.position = reordered.tolist()
        self._joint_command.velocity = np.zeros(8).tolist()
        self._joint_command.effort = np.zeros(8).tolist()
        self._joint_publisher.publish(self._joint_command)

    # --- Observation builders ---

    def _compute_observation(self, joint_state: JointState, imu: Imu | None):
        if self._policy_type == 'jump':
            return self._obs_jump(joint_state, imu)
        return self._obs_locomotion(joint_state, imu)

    def _get_joint_state(self, joint_state: JointState):
        """Extract joint positions and velocities in controller order."""
        pos = np.zeros(8, dtype=np.float32)
        vel = np.zeros(8, dtype=np.float32)
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                if idx < len(joint_state.position):
                    pos[i] = joint_state.position[idx]
                if idx < len(joint_state.velocity):
                    vel[i] = joint_state.velocity[idx]
        return pos, vel

    def _get_imu_derived(self, imu: Imu | None):
        """Extract gravity direction, angular velocity, and update linear velocity estimate."""
        if imu is None:
            gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            ang_vel = np.zeros(3, dtype=np.float32)
            if not self._warned_no_imu_fallback and self._policy_counter > 50:
                self._logger.warn(
                    "No /imu received yet; using fallback gravity=[0,0,-1], ang_vel=[0,0,0]."
                )
                self._warned_no_imu_fallback = True
        else:
            quat = np.array([imu.orientation.w, imu.orientation.x,
                             imu.orientation.y, imu.orientation.z])
            R_BI = self.quat_to_rot_matrix(quat).T
            # Isaac Sim USD stage is Y-up: gravity in stage world frame is [0, -1, 0].
            # R_BI uses this Y-up convention, so [0, -1, 0] correctly projects
            # world gravity into the body frame expected by the policy.
            gravity_b = R_BI @ np.array([0.0, -1.0, 0.0])
            ang_vel_raw = np.array([imu.angular_velocity.x,
                                    imu.angular_velocity.y,
                                    imu.angular_velocity.z])
            # Convert angular velocity from Y-up IMU frame to Z-up training frame
            # via R_x(+90°): [x, y, z] -> [x, -z, y]
            ang_vel = np.array([ang_vel_raw[0], -ang_vel_raw[2], ang_vel_raw[1]],
                               dtype=np.float32)
            # Remap IMU/body axes into the policy body frame.
            gravity_b = self._imu_axis_remap @ gravity_b
            ang_vel = self._imu_axis_remap @ ang_vel

        # If no odometry is available, keep a conservative fallback.
        if not self._has_odom:
            if self._use_cmd_as_lin_vel_when_no_odom and self._policy_type in ('stand', 'walk'):
                cmd = self._get_cmd_vector()
                self._lin_vel_b = np.array(
                    [
                        cmd[0] * self._cmd_vel_to_lin_vel_gain,
                        cmd[1] * self._cmd_vel_to_lin_vel_gain,
                        0.0,
                    ],
                    dtype=np.float32,
                )
                if not self._warned_cmd_vel_fallback and self._policy_counter > 200:
                    self._logger.warn(
                        "No /odom received; using scaled cmd_vel fallback for base_lin_vel observation."
                    )
                    self._warned_cmd_vel_fallback = True
            else:
                self._lin_vel_b = np.zeros(3)

        return gravity_b, ang_vel

    def _obs_locomotion(self, joint_state: JointState, imu: Imu | None):
        """36-dim: lin_vel(3), ang_vel(3), gravity(3), cmd(3), jpos(8), jvel(8), action(8)"""
        gravity_b, ang_vel = self._get_imu_derived(imu)
        jpos, jvel = self._get_joint_state(joint_state)
        cmd = self._get_cmd_vector()
        if (
            not self._warned_zero_cmd
            and self._policy_type in ('stand', 'walk')
            and np.all(np.abs(cmd) < 1e-5)
            and self._policy_counter > 200
        ):
            self._logger.warn(
                "cmd_vel is still near zero. "
                "These stand/walk policies were trained with mostly non-zero velocity commands."
            )
            self._warned_zero_cmd = True
        if (
            not self._warned_no_odom
            and self._policy_type in ('stand', 'walk')
            and not self._has_odom
            and not self._use_cmd_as_lin_vel_when_no_odom
            and self._policy_counter > 200
        ):
            self._logger.warn(
                "No /odom received yet. base_lin_vel observation stays zero, "
                "which can hurt stand/walk policy stability."
            )
            self._warned_no_odom = True

        obs = np.zeros(36)
        # Base linear velocity.
        obs[0:3] = self._lin_vel_b
        # Base angular velocity.
        obs[3:6] = ang_vel
        # Gravity direction in body frame.
        obs[6:9] = gravity_b
        # Commanded velocity target.
        obs[9:12] = cmd
        # Joint position relative to default pose.
        obs[12:20] = (jpos - self.default_pos) * self._joint_signs
        # Joint velocity in policy frame.
        obs[20:28] = jvel * self._joint_signs
        # Previous policy action.
        obs[28:36] = self._previous_action
        return obs

    def _resolve_training_joint_order(self, order_key: str) -> list[str]:
        if order_key in TRAINING_JOINT_ORDERS:
            return TRAINING_JOINT_ORDERS[order_key].copy()
        self._logger.warn(
            f"Unknown training_joint_order='{order_key}', falling back to 'right_left'."
        )
        self._training_joint_order_key = 'right_left'
        return TRAINING_JOINT_ORDERS['right_left'].copy()

    def _refresh_isaac_joint_order(self, isaac_order: list[str]) -> None:
        filtered_order = [name for name in isaac_order if name in self.joint_names]
        missing = [name for name in self.joint_names if name not in filtered_order]
        if missing:
            self._logger.error(
                f"Missing joints in /joint_states for mapping: {missing}. "
                f"Received: {isaac_order}"
            )
            return
        if self._training_joint_order_key == 'isaac':
            self.joint_names = filtered_order.copy()
            self._joint_limits = np.array(
                [JOINT_LIMITS_BY_NAME[n] for n in self.joint_names],
                dtype=np.float32,
            )
        self.isaac_joint_names = filtered_order
        self._to_isaac_idx = [self.isaac_joint_names.index(n) for n in self.joint_names]

    def _obs_jump(self, joint_state: JointState, imu: Imu | None):
        """191-dim: gravity(3), target_pos_b(3), time_to_go(1), jpos(8), jvel(8), action(8), height_scan(160)"""
        gravity_b, _ = self._get_imu_derived(imu)
        jpos, jvel = self._get_joint_state(joint_state)

        obs = np.zeros(191)
        # Gravity direction in body frame.
        obs[0:3] = gravity_b
        # Jump target in body frame.
        obs[3:6] = self._jump_target
        # Time remaining before takeoff.
        obs[6] = self._jump_time_to_go
        # Joint position relative to default pose.
        obs[7:15] = (jpos - self.default_pos) * self._joint_signs
        # Joint velocity in policy frame.
        obs[15:23] = jvel * self._joint_signs
        # Previous policy action.
        obs[23:31] = self._previous_action
        # Local terrain height samples.
        obs[31:191] = self._height_scan
        return obs

    # --- Inference ---

    def _compute_action(self, obs):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs_t).detach().view(-1).numpy()
        return action

    # --- Utilities ---

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
        ), dtype=np.float64)

    def _parse_axis_remap(self, spec: str) -> np.ndarray:
        """Parse axis remap like 'x,y,z' or 'z,y,-x'."""
        axes = [token.strip().lower() for token in spec.split(',')]
        if len(axes) != 3:
            self._logger.warn(
                f"imu_axis_remap='{spec}' is invalid. Falling back to identity remap."
            )
            return np.eye(3, dtype=np.float32)

        matrix = np.zeros((3, 3), dtype=np.float32)
        used_axes: set[int] = set()
        axis_map = {'x': 0, 'y': 1, 'z': 2}

        for row, token in enumerate(axes):
            sign = -1.0 if token.startswith('-') else 1.0
            axis_name = token[1:] if token.startswith('-') else token
            if axis_name not in axis_map:
                self._logger.warn(
                    f"imu_axis_remap='{spec}' contains unknown axis '{token}'. "
                    "Falling back to identity remap."
                )
                return np.eye(3, dtype=np.float32)
            axis_idx = axis_map[axis_name]
            if axis_idx in used_axes:
                self._logger.warn(
                    f"imu_axis_remap='{spec}' repeats axis '{axis_name}'. "
                    "Falling back to identity remap."
                )
                return np.eye(3, dtype=np.float32)
            used_axes.add(axis_idx)
            matrix[row, axis_idx] = sign

        return matrix

    def _resolve_policy_path(self, explicit_path: str, policy_type: str) -> str:
        """Find policy file: use explicit path if given, otherwise auto-discover."""
        if explicit_path:
            return explicit_path

        # Walk up from __file__ to find the workspace root containing model/isaaclab.
        # This works whether running from source or from colcon install tree.
        ws_root = self._find_ws_root(os.path.dirname(os.path.abspath(__file__)))
        if not ws_root:
            self._logger.error("Could not locate workspace root (model/isaaclab not found in any parent dir)")
            return ''

        pattern = os.path.join(ws_root, 'model', 'isaaclab', policy_type, '*', 'exported', 'policy.pt')
        matches = sorted(glob.glob(pattern))
        if matches:
            resolved = os.path.realpath(matches[-1])
            self._logger.info(f"Auto-discovered policy: {resolved}")
            return resolved

        self._logger.error(f"No policy found for type '{policy_type}' at {pattern}")
        return ''

    @staticmethod
    def _find_ws_root(start: str) -> str:
        """Walk up the directory tree to find the root that contains model/isaaclab."""
        path = os.path.abspath(start)
        for _ in range(12):
            if os.path.isdir(os.path.join(path, 'model', 'isaaclab')):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        return ''

    def load_policy(self):
        if not self._policy_path:
            self._logger.error("No policy path set")
            self.policy = lambda x: torch.zeros(1, 8)
            return
        try:
            with open(self._policy_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
            self.policy = torch.jit.load(buffer)
            self._logger.info(f"Loaded policy from {self._policy_path}")
        except FileNotFoundError:
            self._logger.error(f"Policy file not found: {self._policy_path}")
            self.policy = lambda x: torch.zeros(1, 8)
        except Exception as e:
            self._logger.error(f"Error loading policy: {e}")
            self.policy = lambda x: torch.zeros(1, 8)

    def _get_stamp_prefix(self) -> str:
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f'[{now}][{now_ros}]'


def main(args=None):
    rclpy.init(args=args)
    node = DodoPolicyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
