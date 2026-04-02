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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for Dodo policy controller."""

    # Declare launch arguments
    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('dodo_controller'),
            'policy',
            'dodo_policy.pt'
        ]),
        description='Path to the trained policy file (.pt)'
    )

    action_scale_arg = DeclareLaunchArgument(
        'action_scale',
        default_value='0.5',
        description='Scaling factor for policy output actions'
    )

    decimation_arg = DeclareLaunchArgument(
        'decimation',
        default_value='4',
        description='Run policy every N ticks (reduces computation)'
    )

    publish_period_ms_arg = DeclareLaunchArgument(
        'publish_period_ms',
        default_value='5',
        description='Publishing period in milliseconds'
    )

    # Create the controller node
    dodo_controller_node = Node(
        package='dodo_controller',
        executable='dodo_policy_node',
        name='dodo_policy_controller',
        output='screen',
        parameters=[{
            'policy_path': LaunchConfiguration('policy_path'),
            'action_scale': LaunchConfiguration('action_scale'),
            'decimation': LaunchConfiguration('decimation'),
            'publish_period_ms': LaunchConfiguration('publish_period_ms'),
            'use_sim_time': True,
        }],
        remappings=[
            # Remap topics if needed for your specific setup
            # Example: ('/cmd_vel', '/dodo/cmd_vel'),
        ]
    )

    return LaunchDescription([
        policy_path_arg,
        action_scale_arg,
        decimation_arg,
        publish_period_ms_arg,
        dodo_controller_node,
    ])
