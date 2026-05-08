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
from nav_msgs.msg import Odometry
from message_filters import Subscriber, TimeSynchronizer
from message_filters import ApproximateTimeSynchronizer

ROBOT_TYPE = "dodo" # go2 or dodo
STATE_SOURCE = "odom"  # "imu", "odom", "mixed"

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
        self.declare_parameter('publish_period_ms', 20) # dt * decimation
        self.declare_parameter('policy_path', 'policy/dodo_policy.pt')
        self.declare_parameter('action_scale', 0.2)  # Scale factor for policy output
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
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, # was KEEP_ALL
            depth=10,
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
        # Setup synchronized subscribers for Odometry and joint state data
        self._odom_sub_filter = Subscriber(
            self,
            Odometry,
            'odom',
            qos_profile=sim_qos_profile,
        )
        self._joint_states_sub_filter = Subscriber(
            self,
            JointState,
            'joint_states',
            qos_profile=sim_qos_profile,
        )
        queue_size = 10

        if STATE_SOURCE == "imu":
            subscribers = [self._joint_states_sub_filter, self._imu_sub_filter]
        elif STATE_SOURCE == "odom":
            subscribers = [self._joint_states_sub_filter, self._odom_sub_filter]
        elif STATE_SOURCE == "mixed":
            subscribers = [
                self._joint_states_sub_filter,
                self._odom_sub_filter,
                self._imu_sub_filter
            ]

        # Time synchronizer to ensure joint state and IMU data are processed together
        #self.sync = TimeSynchronizer(subscribers, queue_size)
        self.sync = ApproximateTimeSynchronizer(
            subscribers,
            queue_size=queue_size,
            slop=0.005,
        )

        self.sync.registerCallback(self._tick)

        self._episode_time = 0.0  # Track episode time for optional clock observation

        # Initialize state variables
        self._joint_state = JointState()
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        #self._imu = Imu()
        self._policy_counter = 0
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self._lin_vel_b = np.zeros(3)  # Linear velocity in body frame
        self._dt = 0.0  # Time delta between ticks

        # set up initial cmd velocity for testing
        self._cmd_vel.linear.x = 0.4
        self._cmd_vel.linear.y = 0.0
        self._cmd_vel.angular.z = 0.0

        # TODO: Update default joint positions for Dodo's nominal stance
        # Currently set to zeros - should be updated based on your trained policy
        self.default_pos = np.array([
            0.0, # hip_right (hip)
            0.43, # upper_leg_right (thigh)
            -1.0, # lower_leg_right (knee)
            0.57, # foot_right (foot)
            0.0, # hip_left (hip)   
            0.43, # upper_leg_left (thigh)
            -1.0, # lower_leg_left (knee)
            0.57  # foot_left (foot)
        ]) if ROBOT_TYPE == "dodo" else np.array([
            0.1, # FL_hip_joint
            0.8, # FL_thigh_joint
            -1.5, # FL_calf_joint
            -0.1, # FR_hip_joint
            0.8, # FR_thigh_joint
            -1.5, # FR_calf_joint
            0.1, # RL_hip_joint
            1.0, # RL_thigh_joint
            -1.5, # RL_calf_joint
            -0.1, # RR_hip_joint
            1.0,  # RR_thigh_joint
            -1.5 # RR_calf_joint
        ])

        # Joint names in the order expected by the policy
        # Based on joint_names that were used during training (e.g. in Isaac Sim or Genesis) - we will have to remap those later to the order coming from the JointState messages, which may differ.
        self.joint_names_train = [
            'hip_right',
            'upper_leg_right',
            'lower_leg_right',
            'foot_right',
            'hip_left',
            'upper_leg_left',
            'lower_leg_left',
            'foot_left',
        ] if ROBOT_TYPE == "dodo" else [
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint'
        ]

        self._previous_action = np.zeros(len(self.joint_names_train))
        self.action = np.zeros(len(self.joint_names_train))

        # Load neural network policy
        self.policy_path = self.get_parameter('policy_path').value
        self.load_policy()

        self.joint_names_sim = None

        self._sim_to_train_idx = None   # read JointState in train/policy order
        self._train_to_sim_idx = None   # publish action/default_pos in sim order
        self._joint_state_name_tuple = None

        self._logger.info("Initializing DodoPolicyController")
        self._logger.info(f"Policy path: {self.policy_path}")
        self._logger.info(f"Action scale: {self._action_scale}")
        self._logger.info(f"Decimation: {self._decimation}")

    def _cmd_vel_callback(self, msg):
        """Store the latest velocity command."""
        self._cmd_vel = msg

    def _tick(self, joint_state: JointState, *state_msgs):
        """Process synchronized joint state and IMU data to generate robot commands.

        This method is called whenever new joint state and IMU data are available.
        It computes the policy's action and publishes the resulting joint commands.

        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data (orientation, angular velocity, acceleration)
        """
        imu = None
        odom = None

        if STATE_SOURCE == "imu":
            imu = state_msgs[0]
        elif STATE_SOURCE == "odom":
            odom = state_msgs[0]
        elif STATE_SOURCE == "mixed":
            odom = state_msgs[0]
            imu = state_msgs[1]
    
        # Reset if time jumped backwards (most likely due to sim time reset)
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_tick_time:
            self._logger.error(
                f'{self._get_stamp_prefix()} Time jumped backwards. Resetting.'
            )
            self._episode_time = 0.0
            self._lin_vel_b = np.zeros(3)
            self._policy_counter = 0
            self._previous_action = np.zeros(len(self.joint_names_train))
            self._last_tick_time = now
            return

        # Calculate time delta since last tick
        self._dt = (now - self._last_tick_time)
        self._last_tick_time = now
        self._episode_time += self._dt

        # Run the control policy
        if self._episode_time < 0.1:
            self.forward(joint_state, odom, imu)
            self.action[:] = 0.0
        else:
            self.forward(joint_state, odom, imu)
        #self.forward(joint_state, odom, imu)

        # Prepare and publish the joint command message
        self._joint_command.header.stamp = self.get_clock().now().to_msg()

        # Compute final joint positions by adding scaled actions to default positions
        action_pos = np.clip(self.action, -10.0, 10.0) * self._action_scale + self.default_pos
        #action_pos = self.default_pos # TODO for evaluation just publish default pose

        # convert to IsaacSim order for execution
        target_pos_sim = self._to_sim_order(action_pos)

        self._joint_command.name = self.joint_names_sim
        self._joint_command.position = target_pos_sim.tolist()
        self._joint_command.velocity = []
        self._joint_command.effort = []

        self._joint_publisher.publish(self._joint_command)

        #if self._policy_counter % 50 == 0:
        #    self._logger.info(f"tick dt={self._dt:.5f}, policy_freq≈{1.0/(self._dt*self._decimation):.1f} Hz")

    def _compute_observation(self, joint_state: JointState, odom: Odometry, imu: Imu):
        """Compute the policy observation vector from robot state.
        
        => Observations differ between policies trained in isaacsim and genesis.
           The Observations that you want to use as model input may from person to person.
           As the Team training on genesis decided on a different observation space than the one used for training in Isaac Sim, 
           we provide this separate method to compute the observation vector in the format expected by the genesis-trained policy. 
           You can modify this method to match the observation space you used for your PPO training.

        TODO: Update observation dimensions based on your PPO training configuration.
        Current structure follows a typical quadruped observation space:
        - Linear velocity (body frame): 3
        - Angular velocity (body frame): 3
        - Gravity direction (body frame): 3
        - Joint positions (relative to default): 8
            model input order in genesis is:
                - hip_right (hip)
                - upper_leg_right (thigh)
                - lower_leg_right (knee)
                - foot_right (foot)
                - hip_left (hip)
                - upper_leg_left (thigh)
                - lower_leg_left (knee)
                - foot_left (foot)
        - Joint velocities: 8
        - Last action: 8
        - Command velocity: 3
        - clock (sin / cos): 2 -> This observation is optional and can be removed if not used in your training.
        Total: 37 dimensions or 39 if including clock observation

        Args:
            joint_state: Current joint positions and velocities
            odom: Current odometry data

        Returns:
            np.ndarray: Observation vector for the policy
        """

        use_clock_obs = False # Set to True if you included a clock observation in your training

        observation_scales = { # TODO use the scales that you used during training for consistency.
            'ang_vel': 1.0,  # Scale angular velocity if needed
            'dof_pos': 1.0,  # Scale joint positions if needed
            'dof_vel': 1.0,  # Scale joint velocities if needed
            'lin_vel': 1.0,  # Scale command velocities if needed
        }

        # Extract quaternion orientation from Odeometry
        if STATE_SOURCE == "odom":
            quat_I = odom.pose.pose.orientation

            lin_vel_obs = np.array([
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
                odom.twist.twist.linear.z,
            ])

            ang_vel_obs = np.array([
                odom.twist.twist.angular.x,
                odom.twist.twist.angular.y,
                odom.twist.twist.angular.z,
            ])

        elif STATE_SOURCE == "mixed":
            quat_I = imu.orientation

            lin_vel_obs = np.array([
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
                odom.twist.twist.linear.z,
            ])

            ang_vel_obs = np.array([
                odom.twist.twist.angular.x,
                odom.twist.twist.angular.y,
                odom.twist.twist.angular.z,
            ])

        elif STATE_SOURCE == "imu":
            quat_I = imu.orientation

            lin_vel_obs = np.zeros(3)  # besser als integrierte IMU-Acceleration
            ang_vel_obs = np.array([
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ])

        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])
        R_BI = self.quat_to_rot_matrix(quat_array).T
        # Calculate gravity direction in body frame
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        # Initialize observation vector (36-dim for typical quadruped)
        obs_count = 3 + 3 + 3 + 3 + 3*len(self.joint_names_train)
        obs = np.zeros(obs_count + 2) if use_clock_obs else np.zeros(obs_count)

        # Fill observation vector components:
        # Base linear velocity (3)
        obs[0:3] = lin_vel_obs * observation_scales["lin_vel"]
        
        # Base angular velocity (3)
        obs[3:6] = ang_vel_obs * observation_scales["ang_vel"]

        # Gravity direction (3)
        obs[6:9] = gravity_b 

        # Map joint states from message to our ordered arrays
        joint_pos, joint_vel = self._extract_joint_state_train_order(joint_state)

        # Velocity commands (3)
        cmd_vel = [
            self._cmd_vel.linear.x * observation_scales['lin_vel'],
            self._cmd_vel.linear.y * observation_scales['lin_vel'],
            self._cmd_vel.angular.z * observation_scales['ang_vel']
        ]
        obs[9:12] = np.array(cmd_vel)
        new_start_idx = 12

        # Store joint positions 
        obs[new_start_idx:new_start_idx + len(self.joint_names_train)] = (joint_pos - self.default_pos) * observation_scales['dof_pos']
        #obs[9:17] = 0.0
        new_start_idx += len(self.joint_names_train)

        # Store joint velocities
        obs[new_start_idx:new_start_idx + len(self.joint_names_train)] = joint_vel * observation_scales['dof_vel']
        new_start_idx += len(self.joint_names_train)

        # Store previous actions
        obs[new_start_idx:new_start_idx + len(self.joint_names_train)] = self._previous_action #* observation_scales['dof_pos'] # we did not scale it again in genesis
        new_start_idx += len(self.joint_names_train)
        
        # Store clock observation (optional)
        if use_clock_obs:
            period = 1.2
            phase = (self._episode_time % period) / period
            obs[obs_count:obs_count + 2] = [
                np.sin(2.0 * np.pi * phase),
                np.cos(2.0 * np.pi * phase)
            ]

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        if self._policy_counter % 50 == 0:
            self._logger.info(f"cmd: {[self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z]}")
            self._logger.info(f"lin_vel_obs: {lin_vel_obs}")
            self._logger.info(f"ang_vel_obs: {ang_vel_obs}")
            self._logger.info(f"gravity_b: {gravity_b}")
            self._logger.info(f"joint_rel min/max: {(joint_pos - self.default_pos).min():.3f}, {(joint_pos - self.default_pos).max():.3f}")
            self._logger.info(f"action min/max: {self.action.min():.3f}, {self.action.max():.3f}")

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

    def forward(self, joint_state: JointState, odom: Odometry, imu: Imu):
        """Process sensor data and compute control actions.

        This combines observation computation and policy evaluation.
        The policy is run at a reduced rate (decimation) to save computation.

        Args:
            joint_state: Current joint positions and velocities
            odom: Current odometry data
            imu: Current IMU data
        """
        # Compute observation from current state
        obs = self._compute_observation(joint_state, odom, imu)

        # Run policy at reduced frequency (every _decimation ticks)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            self.action = np.clip(self.action, -10, 10)

        # previous_action sollte die zuletzt publizierte/gehaltene action sein
        self._previous_action = self.action.copy()

        # if self._policy_counter % 50 == 0:
        #     self._logger.info(f"action raw/clipped: {self.action}")
        #     self._logger.info(f"target train order: {self.action * self._action_scale + self.default_pos}")

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
            self.policy = lambda x: torch.zeros(1, len(self.joint_names_train))
        except Exception as e:
            self._logger.error(f"Error loading policy: {e}")
            raise

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
    
    def _build_joint_remap(self, joint_state: JointState):
        """Build and cache joint remapping between IsaacSim order and policy/train order."""
        incoming_names = tuple(joint_state.name)

        if (
            self._sim_to_train_idx is not None
            and self._train_to_sim_idx is not None
            and incoming_names == self._joint_state_name_tuple
        ):
            return

        self.joint_names_sim = list(joint_state.name)
        self._joint_state_name_tuple = incoming_names

        sim_name_to_idx = {name: i for i, name in enumerate(self.joint_names_sim)}
        train_name_to_idx = {name: i for i, name in enumerate(self.joint_names_train)}

        missing_in_sim = [name for name in self.joint_names_train if name not in sim_name_to_idx]
        if missing_in_sim:
            raise RuntimeError(
                f"JointState missing train joints: {missing_in_sim}. "
                f"Received joints: {self.joint_names_sim}"
            )

        extra_in_sim = [name for name in self.joint_names_sim if name not in train_name_to_idx]
        if extra_in_sim:
            raise RuntimeError(
                f"JointState contains extra joints not used by policy: {extra_in_sim}. "
                f"Received joints: {self.joint_names_sim}"
            )

        self._sim_to_train_idx = np.array(
            [sim_name_to_idx[name] for name in self.joint_names_train],
            dtype=np.int64,
        )

        self._train_to_sim_idx = np.array(
            [train_name_to_idx[name] for name in self.joint_names_sim],
            dtype=np.int64,
        )

        self._logger.info(f"Built joint remap once.")
        self._logger.info(f"Train/policy order: {self.joint_names_train}")
        self._logger.info(f"Sim/Isaac order:   {self.joint_names_sim}")
        self._logger.info(f"sim_to_train_idx: {self._sim_to_train_idx.tolist()}")
        self._logger.info(f"train_to_sim_idx: {self._train_to_sim_idx.tolist()}")


    def _extract_joint_state_train_order(self, joint_state: JointState):
        """Return joint positions/velocities in train/policy order."""
        self._build_joint_remap(joint_state)

        joint_pos_all = np.asarray(joint_state.position, dtype=np.float64)

        if len(joint_state.velocity) == len(joint_state.name):
            joint_vel_all = np.asarray(joint_state.velocity, dtype=np.float64)
        else:
            joint_vel_all = np.zeros(len(joint_state.name), dtype=np.float64)

        joint_pos_train = joint_pos_all[self._sim_to_train_idx]
        joint_vel_train = joint_vel_all[self._sim_to_train_idx]

        return joint_pos_train, joint_vel_train
    
    def _to_sim_order(self, values_train_order: np.ndarray):
        """Convert an array from train/policy order to IsaacSim JointState order."""
        if self._train_to_sim_idx is None:
            raise RuntimeError("Joint remap has not been built yet.")

        return np.asarray(values_train_order, dtype=np.float64)[self._train_to_sim_idx]


def main(args=None):
    """Main function to initialize and run the Dodo policy controller node."""
    rclpy.init(args=args)
    node = DodoPolicyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
