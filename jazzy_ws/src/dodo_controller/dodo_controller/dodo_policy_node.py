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

import rclpy
import torch
import numpy as np
import io
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from message_filters import Subscriber, TimeSynchronizer


class DodoPolicyController(Node):
    """PPO policy controller for Dodo quadruped robot.

    This ROS 2 node subscribes to velocity commands and synchronized joint/IMU
    data, processes the data through a trained PPO policy network, and publishes
    joint commands for controlling the Dodo robot's movements.
    """

    def __init__(self):
        """Initialize the Dodo policy controller node."""
        super().__init__('dodo_policy_controller')

        # Declare and set parameters
        self.declare_parameter('publish_period_ms', 5)
        self.declare_parameter('policy_path', 'policy/dodo_policy.pt')
        self.declare_parameter('action_scale', 0.5)  # Scale factor for policy output
        self.declare_parameter('decimation', 4)  # Run policy every N ticks
        self.set_parameters(
            [rclpy.parameter.Parameter(
                'use_sim_time',
                rclpy.Parameter.Type.BOOL,
                True
            )]
        )

        self._logger = self.get_logger()

        # Get parameters
        self._action_scale = self.get_parameter('action_scale').value
        self._decimation = self.get_parameter('decimation').value

        # Configure QoS profile for simulation
        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        # Create subscription for velocity commands
        self._cmd_vel_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self._cmd_vel_callback,
            qos_profile=10)

        # Create publisher for joint commands
        self._joint_publisher = self.create_publisher(
            JointState,
            'joint_command',
            qos_profile=sim_qos_profile)

        # Setup synchronized subscribers for IMU and joint state data
        self._imu_sub_filter = Subscriber(
            self,
            Imu,
            'imu',
            qos_profile=sim_qos_profile,
        )
        self._joint_states_sub_filter = Subscriber(
            self,
            JointState,
            'joint_states',
            qos_profile=sim_qos_profile,
        )
        queue_size = 10
        subscribers = [self._joint_states_sub_filter, self._imu_sub_filter]

        # Time synchronizer to ensure joint state and IMU data are processed together
        self.sync = TimeSynchronizer(subscribers, queue_size)
        self.sync.registerCallback(self._tick)

        # Load neural network policy
        self.policy_path = self.get_parameter('policy_path').value
        self.load_policy()

        # Initialize state variables
        self._joint_state = JointState()
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._imu = Imu()
        self._previous_action = np.zeros(8)  # 8 joints for Dodo
        self._policy_counter = 0
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self._lin_vel_b = np.zeros(3)  # Linear velocity in body frame
        self._dt = 0.0  # Time delta between ticks

        # TODO: Update default joint positions for Dodo's nominal stance
        # Currently set to zeros - should be updated based on your trained policy
        self.default_pos = np.array([
            0.0, 0.0, 0.0, 0.0,  # right leg (hip, knee, etc.)
            0.0, 0.0, 0.0, 0.0   # left leg
        ])

        # Joint names in the order expected by the policy
        # Based on joint_names_dodobot_v3.yaml (excluding empty string at index 0)
        self.joint_names = [
            'right_joint_1',
            'right_joint_2',
            'right_joint_3',
            'right_joint_4',
            'left_joint_1',
            'left_joint_2',
            'left_joint_3',
            'left_joint_4'
        ]

        self._logger.info("Initializing DodoPolicyController")
        self._logger.info(f"Policy path: {self.policy_path}")
        self._logger.info(f"Action scale: {self._action_scale}")
        self._logger.info(f"Decimation: {self._decimation}")

    def _cmd_vel_callback(self, msg):
        """Store the latest velocity command."""
        self._cmd_vel = msg

    def _tick(self, joint_state: JointState, imu: Imu):
        """Process synchronized joint state and IMU data to generate robot commands.

        This method is called whenever new joint state and IMU data are available.
        It computes the policy's action and publishes the resulting joint commands.

        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data (orientation, angular velocity, acceleration)
        """
        # Reset if time jumped backwards (most likely due to sim time reset)
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_tick_time:
            self._logger.error(
                f'{self._get_stamp_prefix()} Time jumped backwards. Resetting.'
            )

        # Calculate time delta since last tick
        self._dt = (now - self._last_tick_time)
        self._last_tick_time = now

        # Run the control policy
        self.forward(joint_state, imu)

        # Prepare and publish the joint command message
        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.joint_names

        # Compute final joint positions by adding scaled actions to default positions
        action_pos = self.default_pos + self.action * self._action_scale
        self._joint_command.position = action_pos.tolist()
        self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
        self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
        self._joint_publisher.publish(self._joint_command)

    def _compute_observation(self, joint_state: JointState, imu: Imu):
        """Compute the policy observation vector from robot state.

        TODO: Update observation dimensions based on your PPO training configuration.
        Current structure follows a typical quadruped observation space:
        - Linear velocity (body frame): 3
        - Angular velocity (body frame): 3
        - Gravity direction (body frame): 3
        - Command velocity: 3
        - Joint positions (relative to default): 8
        - Joint velocities: 8
        - Previous action: 8
        Total: 36 dimensions

        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data

        Returns:
            np.ndarray: Observation vector for the policy
        """
        # Extract quaternion orientation from IMU
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])

        # Convert quaternion to rotation matrix (transpose for body to inertial frame)
        R_BI = self.quat_to_rot_matrix(quat_array).T

        # Extract linear acceleration and integrate to estimate velocity
        lin_acc_b = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z
        ])

        # Simple integration to estimate velocity
        self._lin_vel_b = lin_acc_b * self._dt + self._lin_vel_b

        # Extract angular velocity
        ang_vel_b = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z
        ])

        # Calculate gravity direction in body frame
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        # Initialize observation vector (36-dim for typical quadruped)
        obs = np.zeros(36)

        # Fill observation vector components:
        # Base linear velocity (3)
        obs[:3] = self._lin_vel_b

        # Base angular velocity (3)
        obs[3:6] = ang_vel_b

        # Gravity direction (3)
        obs[6:9] = gravity_b

        # Velocity commands (3)
        cmd_vel = [
            self._cmd_vel.linear.x,
            self._cmd_vel.linear.y,
            self._cmd_vel.angular.z
        ]
        obs[9:12] = np.array(cmd_vel)

        # Joint states (8 positions + 8 velocities)
        current_joint_pos = np.zeros(8)
        current_joint_vel = np.zeros(8)

        # Map joint states from message to our ordered arrays
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]

        # Store joint positions relative to default pose
        obs[12:20] = current_joint_pos - self.default_pos

        # Store joint velocities
        obs[20:28] = current_joint_vel

        # Store previous actions
        obs[28:36] = self._previous_action

        return obs

    def _compute_action(self, obs):
        """Run the neural network policy to compute an action from the observation.

        Args:
            obs: Observation vector containing robot state information

        Returns:
            np.ndarray: Action vector containing joint position adjustments
        """
        # Run inference with the PyTorch policy
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        """Process sensor data and compute control actions.

        This combines observation computation and policy evaluation.
        The policy is run at a reduced rate (decimation) to save computation.

        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data
        """
        # Compute observation from current state
        obs = self._compute_observation(joint_state, imu)

        # Run policy at reduced frequency (every _decimation ticks)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        self._policy_counter += 1

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert input quaternion to rotation matrix.

        Args:
            quat (np.ndarray): Input quaternion (w, x, y, z).

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

    def load_policy(self):
        """Load the neural network policy from the specified path."""
        try:
            # Load policy from file to io.BytesIO object
            with open(self.policy_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
            # Load TorchScript model from buffer
            self.policy = torch.jit.load(buffer)
            self._logger.info(f"Successfully loaded policy from {self.policy_path}")
        except FileNotFoundError:
            self._logger.error(f"Policy file not found: {self.policy_path}")
            self._logger.warn("Please place your trained policy at the specified path")
            # Create a dummy policy for testing
            self.policy = lambda x: torch.zeros(1, 8)
        except Exception as e:
            self._logger.error(f"Error loading policy: {e}")
            self.policy = lambda x: torch.zeros(1, 8)

    def _get_stamp_prefix(self) -> str:
        """Create a timestamp prefix for logging with both system and ROS time.

        Returns:
            str: Formatted timestamp string with system and ROS time
        """
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f'[{now}][{now_ros}]'

    def header_time_in_seconds(self, header) -> float:
        """Convert a ROS message header timestamp to seconds.

        Args:
            header: ROS message header containing timestamp

        Returns:
            float: Time in seconds
        """
        return header.stamp.sec + header.stamp.nanosec * 1e-9


def main(args=None):
    """Main function to initialize and run the Dodo policy controller node."""
    rclpy.init(args=args)
    node = DodoPolicyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
