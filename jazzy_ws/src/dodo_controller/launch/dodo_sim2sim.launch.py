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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    """Generate launch description for Dodo sim2sim (Isaac Sim + controller)."""

    # Declare launch arguments
    scene_path_arg = DeclareLaunchArgument(
        'scene_path',
        default_value='',
        description='Path to Isaac Sim USD scene file'
    )

    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('dodo_controller'),
            'policy',
            'dodo_policy.pt'
        ]),
        description='Path to the trained policy file (.pt)'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Isaac Sim in headless mode'
    )

    # Isaac Sim launcher node
    isaacsim_node = Node(
        package='isaacsim',
        executable='isaac_sim_launcher',
        name='isaac_sim',
        output='screen',
        parameters=[{
            'gui': LaunchConfiguration('scene_path'),
            'headless': LaunchConfiguration('headless'),
            'use_sim_time': True,
        }]
    )

    # Dodo controller node
    dodo_controller_node = Node(
        package='dodo_controller',
        executable='dodo_policy_node',
        name='dodo_policy_controller',
        output='screen',
        parameters=[{
            'policy_path': LaunchConfiguration('policy_path'),
            'action_scale': 0.5,
            'decimation': 4,
            'publish_period_ms': 5,
            'use_sim_time': True,
        }]
    )

    return LaunchDescription([
        scene_path_arg,
        policy_path_arg,
        headless_arg,
        isaacsim_node,
        dodo_controller_node,
    ])
