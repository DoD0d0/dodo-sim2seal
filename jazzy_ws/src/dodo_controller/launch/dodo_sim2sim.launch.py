#!/usr/bin/env python3

import os
import glob

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    policy_path = LaunchConfiguration('policy_path').perform(context)
    policy_type = LaunchConfiguration('policy_type').perform(context)
    scene_path = LaunchConfiguration('scene_path').perform(context)
    training_joint_order = LaunchConfiguration('training_joint_order').perform(context)
    base_height = LaunchConfiguration('base_height').perform(context)
    joint_stiffness = LaunchConfiguration('joint_stiffness').perform(context)
    joint_damping = LaunchConfiguration('joint_damping').perform(context)
    max_policy_action_abs = float(LaunchConfiguration('max_policy_action_abs').perform(context))
    max_action_delta = float(LaunchConfiguration('max_action_delta').perform(context))
    startup_ramp_sec = float(LaunchConfiguration('startup_ramp_sec').perform(context))
    hold_default_pose_sec = float(LaunchConfiguration('hold_default_pose_sec').perform(context))
    use_cmd_as_lin_vel_when_no_odom = LaunchConfiguration(
        'use_cmd_as_lin_vel_when_no_odom'
    ).perform(context)
    use_first_joint_state_as_default_pos = LaunchConfiguration(
        'use_first_joint_state_as_default_pos'
    ).perform(context)
    use_steady_time_for_control = LaunchConfiguration(
        'use_steady_time_for_control'
    ).perform(context)
    wait_for_cmd_vel = LaunchConfiguration('wait_for_cmd_vel').perform(context)
    cmd_vel_to_lin_vel_gain = float(LaunchConfiguration('cmd_vel_to_lin_vel_gain').perform(context))
    joint_signs = LaunchConfiguration('joint_signs').perform(context)

    # Auto-discover policy if not given
    if not policy_path:
        ws_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
        pattern = os.path.join(ws_root, 'model', 'isaaclab', policy_type, '*', 'exported', 'policy.pt')
        matches = sorted(glob.glob(pattern))
        policy_path = os.path.realpath(matches[-1]) if matches else ''

    # Resolve open_dodo_stage.py path
    isaacsim_scripts = os.path.join(
        os.path.dirname(__file__), '..', '..', 'isaacsim', 'scripts')
    stage_script = os.path.realpath(
        os.path.join(isaacsim_scripts, 'open_dodo_stage.py'))

    # Isaac Sim launcher — uses open_dodo_stage.py for stage init
    isaacsim_node = Node(
        package='isaacsim',
        executable='run_isaacsim.py',
        name='isaac_sim',
        output='screen',
        parameters=[{
            'version': LaunchConfiguration('version'),
            'gui': scene_path,
            'stage_script': stage_script,
            'stage_script_args': (
                f'--base-height {base_height} '
                f'--joint-stiffness {joint_stiffness} '
                f'--joint-damping {joint_damping}'
            ),
            'play_sim_on_start': True,
            'use_internal_libs': True,
            'headless': LaunchConfiguration('headless'),
            'use_sim_time': True,
        }]
    )

    # Controller node
    controller_node = Node(
        package='dodo_controller',
        executable='dodo_policy_node',
        name='dodo_policy_controller',
        output='screen',
        parameters=[{
            'policy_path': policy_path,
            'policy_type': policy_type,
            'action_scale': -1.0,  # auto per policy_type
            'decimation': 4,
            'publish_period_ms': 5,
            'training_joint_order': training_joint_order,
            'max_policy_action_abs': max_policy_action_abs,
            'max_action_delta': max_action_delta,
            'startup_ramp_sec': startup_ramp_sec,
            'hold_default_pose_sec': hold_default_pose_sec,
            'use_cmd_as_lin_vel_when_no_odom': use_cmd_as_lin_vel_when_no_odom,
            'use_first_joint_state_as_default_pos': use_first_joint_state_as_default_pos,
            'use_steady_time_for_control': use_steady_time_for_control,
            'wait_for_cmd_vel': wait_for_cmd_vel,
            'cmd_vel_to_lin_vel_gain': cmd_vel_to_lin_vel_gain,
            'joint_signs': joint_signs,
            'use_sim_time': True,
        }]
    )

    return [isaacsim_node, controller_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene_path', default_value='',
                              description='Path to USD scene file'),
        DeclareLaunchArgument('policy_path', default_value='',
                              description='Explicit policy .pt path (auto-discovers if empty)'),
        DeclareLaunchArgument('policy_type', default_value='stand',
                              description='Policy type: stand, walk, jump'),
        DeclareLaunchArgument('version', default_value='5.1.0',
                              description='Isaac Sim version'),
        DeclareLaunchArgument('headless', default_value='',
                              description='Headless mode: native, webrtc, or empty for GUI'),
        DeclareLaunchArgument('base_height', default_value='0.575',
                              description='Initial robot base height used by open_dodo_stage.py'),
        DeclareLaunchArgument('joint_stiffness', default_value='40.0',
                              description='Joint stiffness passed to open_dodo_stage.py'),
        DeclareLaunchArgument('joint_damping', default_value='2.0',
                              description='Joint damping passed to open_dodo_stage.py'),
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
        OpaqueFunction(function=launch_setup),
    ])
