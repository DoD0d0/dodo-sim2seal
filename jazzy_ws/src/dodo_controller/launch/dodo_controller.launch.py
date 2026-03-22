#!/usr/bin/env python3

import os
import glob

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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


def resolve_policy_path(context):
    policy_path = LaunchConfiguration('policy_path').perform(context)
    policy_type = LaunchConfiguration('policy_type').perform(context)

    # If explicit path given, use it
    if policy_path:
        resolved = policy_path
    else:
        ws_root = _find_ws_root(os.path.dirname(os.path.abspath(__file__)))
        if ws_root:
            pattern = os.path.join(ws_root, 'model', 'isaaclab', policy_type, '*', 'exported', 'policy.pt')
            matches = sorted(glob.glob(pattern))
            resolved = os.path.realpath(matches[-1]) if matches else ''
        else:
            resolved = ''

    return [
        Node(
            package='dodo_controller',
            executable='dodo_policy_node',
            name='dodo_policy_controller',
            output='screen',
            parameters=[{
                'policy_path': resolved,
                'policy_type': policy_type,
                'action_scale': LaunchConfiguration('action_scale'),
                'decimation': LaunchConfiguration('decimation'),
                'publish_period_ms': LaunchConfiguration('publish_period_ms'),
                'training_joint_order': LaunchConfiguration('training_joint_order'),
                'max_policy_action_abs': LaunchConfiguration('max_policy_action_abs'),
                'max_action_delta': LaunchConfiguration('max_action_delta'),
                'startup_ramp_sec': LaunchConfiguration('startup_ramp_sec'),
                'hold_default_pose_sec': LaunchConfiguration('hold_default_pose_sec'),
                'use_cmd_as_lin_vel_when_no_odom': LaunchConfiguration('use_cmd_as_lin_vel_when_no_odom'),
                'use_first_joint_state_as_default_pos': LaunchConfiguration('use_first_joint_state_as_default_pos'),
                'use_steady_time_for_control': LaunchConfiguration('use_steady_time_for_control'),
                'wait_for_cmd_vel': LaunchConfiguration('wait_for_cmd_vel'),
                'cmd_vel_to_lin_vel_gain': LaunchConfiguration('cmd_vel_to_lin_vel_gain'),
                'joint_signs': LaunchConfiguration('joint_signs'),
                'imu_axis_remap': LaunchConfiguration('imu_axis_remap'),
                'use_sim_time': True,
            }],
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('policy_path', default_value='',
                              description='Explicit path to policy .pt file (auto-discovers if empty)'),
        DeclareLaunchArgument('policy_type', default_value='stand',
                              description='Policy type: stand, walk, jump'),
        DeclareLaunchArgument('action_scale', default_value='-1.0',
                              description='Action scale (-1 = auto per policy_type)'),
        DeclareLaunchArgument('decimation', default_value='4'),
        DeclareLaunchArgument('publish_period_ms', default_value='5'),
        DeclareLaunchArgument(
            'training_joint_order',
            default_value='right_left',
            description='Policy training joint order: right_left, left_right, or isaac',
        ),
        DeclareLaunchArgument(
            'max_policy_action_abs',
            default_value='2.0',
            description='Clip raw policy output to [-max_policy_action_abs, +max_policy_action_abs]. '
                        'Set <=0 to disable.',
        ),
        DeclareLaunchArgument(
            'max_action_delta',
            default_value='0.15',
            description='Limit per-policy-step action delta. Set <=0 to disable.',
        ),
        DeclareLaunchArgument(
            'startup_ramp_sec',
            default_value='1.0',
            description='Ramp action contribution from 0 to 1 over this many seconds.',
        ),
        DeclareLaunchArgument(
            'hold_default_pose_sec',
            default_value='0.5',
            description='Hold default joint pose for this many seconds before policy control.',
        ),
        DeclareLaunchArgument(
            'use_cmd_as_lin_vel_when_no_odom',
            default_value='true',
            description='If true and /odom is absent, use cmd_vel as base_lin_vel fallback.',
        ),
        DeclareLaunchArgument(
            'use_first_joint_state_as_default_pos',
            default_value='true',
            description='Capture default joint offset from first /joint_states if explicit default is not set.',
        ),
        DeclareLaunchArgument(
            'use_steady_time_for_control',
            default_value='true',
            description='Use monotonic steady time for control dt/ramp to avoid /clock jump issues.',
        ),
        DeclareLaunchArgument(
            'wait_for_cmd_vel',
            default_value='true',
            description='If true and default cmd fallback is disabled, hold default pose until first /cmd_vel.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_to_lin_vel_gain',
            default_value='0.35',
            description='Scale factor for cmd_vel -> base_lin_vel fallback when /odom is missing.',
        ),
        DeclareLaunchArgument(
            'joint_signs',
            default_value='[]',
            description='Optional 8-element list to multiply policy actions per joint (training joint order).',
        ),
        DeclareLaunchArgument(
            'imu_axis_remap',
            default_value='x,y,z',
            description="Axis remap from IMU/body frame into policy frame, e.g. 'z,y,-x'.",
        ),
        OpaqueFunction(function=resolve_policy_path),
    ])
