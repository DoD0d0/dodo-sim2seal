# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to open a USD stage in Isaac Sim (in standard gui mode). This script along with its arguments are automatically passed to Isaac Sim via the ROS2 launch workflow"""

import carb
import argparse
import omni.usd
import asyncio
import omni.client
import omni.kit.async_engine
import omni.timeline

# imports for controlling the stage and prims after opening the stage (usd)
import omni.kit.app
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import XFormPrim, Articulation
import numpy as np

# ROBOT_PRIM = "/dodobot_v3" # change this to the prim path of your robot in the usd stage
# BASE_POS_INIT = np.array([[0, 0, 0.59]], dtype=np.float32) # change this to the initial position of your robot base in the stage
# BASE_ORI_INIT = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32) # change this to the initial rotation position of robot in base stage
# INIT_Q = {
#         "left_joint_1": 0.0,
#         "right_joint_1": 0.0,
#         "left_joint_2": 0.4,
#         "right_joint_2": 0.4,
#         "left_joint_3": -0.7,
#         "right_joint_3": -0.7,
#         "left_joint_4": 0.3,
#         "right_joint_4": 0.3,
#     } # radiant

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='The path to USD stage')
    
    # Add the --start-on-play option
    # If --start-on-play is specified, it sets the value to True
    parser.add_argument('--start-on-play', action='store_true',
                        help='If present, to true.')
    
    try:
        options = parser.parse_args()
    except Exception as e:
        carb.log_error(str(e))
        return

    omni.kit.async_engine.run_coroutine(open_stage_async(options.path, options.start_on_play))

async def open_stage_async(path: str, start_on_play: bool):
    timeline_interface = None
    #if start_on_play:
    timeline_interface = omni.timeline.get_timeline_interface()
    
    async def _open_stage_internal(path):
        is_stage_with_session = False
        try:
            import omni.kit.usd.layers as layers
            live_session_name = layers.get_live_session_name_from_shared_link(path)
            is_stage_with_session = live_session_name is not None
        except Exception:
            pass

        if is_stage_with_session:
            # Try to open the stage with specified live session.
            (success, error) = await layers.get_live_syncing().open_stage_with_live_session_async(path)
        else:
            # Otherwise, use normal stage open.
            (success, error) = await omni.usd.get_context().open_stage_async(path)
        
        if not success:
            carb.log_error(f"Failed to open stage {path}: {error}.")
        else:
            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()
            #set_dodo_base_position()

            if timeline_interface is not None:
                # await omni.kit.app.get_app().next_update_async()
                # await omni.kit.app.get_app().next_update_async()
                # timeline_interface.play()
                # set_dodo_initial_pose() # set the initial pose of the robot after opening the stage
                # carb.log_info("Stage loaded and simulation is playing.")


                timeline_interface.play()

                #await omni.kit.app.get_app().next_update_async()

                #set_dodo_joint_positions()

                #await omni.kit.app.get_app().next_update_async()

                if not start_on_play:
                    timeline_interface.pause() # stop the timeline to set the initial pose of the robot before starting the simulation


            pass
    result, _ = await omni.client.stat_async(path)
    if result == omni.client.Result.OK:
        await _open_stage_internal(path)

        return

    broken_url = omni.client.break_url(path)
    if broken_url.scheme == 'omniverse':
        # Attempt to connect to nucleus server before opening stage
        try:
            from omni.kit.widget.nucleus_connector import get_nucleus_connector
            nucleus_connector = get_nucleus_connector()
        except Exception:
            carb.log_warn("Open stage: Could not import Nucleus connector.")
            return

        server_url = omni.client.make_url(scheme='omniverse', host=broken_url.host)
        nucleus_connector.connect(
            broken_url.host, server_url,
            on_success_fn=lambda *_: asyncio.ensure_future(_open_stage_internal(path)),
            on_failed_fn=lambda *_: carb.log_error(f"Open stage: Failed to connect to server '{server_url}'.")
        )
    else:
        carb.log_warn(f"Open stage: Could not open non-existent url '{path}'.")

# def set_dodo_base_position():
#     dodo_xform = XFormPrim(prim_paths_expr=ROBOT_PRIM)
#     dodo_xform.set_world_poses(BASE_POS_INIT, BASE_ORI_INIT)

# def set_dodo_joint_positions():
#     robot = Articulation(prim_paths_expr=ROBOT_PRIM)
#     robot.initialize()

#     joint_names = robot.dof_names
#     carb.log_info(f"Articulation DOFs: {joint_names}")

#     q = np.zeros((1, len(joint_names)), dtype=np.float32)
#     for i, name in enumerate(joint_names):
#         if name in INIT_Q:
#             q[0, i] = INIT_Q[name]
#         else:
#             carb.log_warn(f"No initial value provided for joint '{name}', using 0.0")

#     qd = np.zeros_like(q)

#     robot.set_joint_positions(q)
#     robot.set_joint_velocities(qd)

#     # nur für spätere resets nützlich
#     robot.set_joints_default_state(positions=q, velocities=qd)

#     carb.log_info(f"Initial joint pose applied: {q}")

    
main()


# The code below is basically the same as the one above but its introducing callbacks to set the initial pose of the robot every time the timeline is played. 
# This can be used to reset the defined initial joint positions even after restarting the simulation in the GUI

# import carb
# import argparse
# import omni.usd
# import asyncio
# import omni.client
# import omni.kit.async_engine
# import omni.timeline
# import omni.kit.app
# import numpy as np

# from isaacsim.core.prims import XFormPrim, Articulation

# ROBOT_PRIM = "/dodobot_v3"
# BASE_POS_INIT = np.array([[0, 0, 0.59]], dtype=np.float32)
# BASE_ORI_INIT = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

# INIT_Q = {
#     "left_joint_1": 0.0,
#     "right_joint_1": 0.0,
#     "left_joint_2": 0.4,
#     "right_joint_2": 0.4,
#     "left_joint_3": -0.7,
#     "right_joint_3": -0.7,
#     "left_joint_4": 0.3,
#     "right_joint_4": 0.3,
# }

# timeline_sub = None
# _is_applying_pose = False


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", type=str, required=True, help="The path to USD stage")
#     parser.add_argument("--start-on-play", action="store_true", help="If present, start sim after loading")
#     options = parser.parse_args()

#     omni.kit.async_engine.run_coroutine(open_stage_async(options.path, options.start_on_play))


# async def open_stage_async(path: str, start_on_play: bool):
#     timeline = omni.timeline.get_timeline_interface()

#     async def _open_stage_internal(path):
#         success, error = await omni.usd.get_context().open_stage_async(path)
#         if not success:
#             carb.log_error(f"Failed to open stage {path}: {error}.")
#             return

#         await omni.kit.app.get_app().next_update_async()
#         await omni.kit.app.get_app().next_update_async()

#         # Base direkt setzen, damit der Roboter korrekt platziert ist
#         set_dodo_base_position()

#         # Callback registrieren
#         install_timeline_callback()

#         # Optional direkt starten
#         if start_on_play:
#             timeline.play()

#     result, _ = await omni.client.stat_async(path)
#     if result == omni.client.Result.OK:
#         await _open_stage_internal(path)
#     else:
#         carb.log_warn(f"Open stage: Could not open non-existent url '{path}'.")


# def install_timeline_callback():
#     global timeline_sub
#     timeline = omni.timeline.get_timeline_interface()

#     def on_timeline_event(event):
#         event_type = int(event.type)

#         if event_type == omni.timeline.TimelineEventType.PLAY.value:
#             carb.log_info("PLAY event detected -> applying initial robot pose")
#             omni.kit.async_engine.run_coroutine(apply_initial_pose_async())

#         elif event_type == omni.timeline.TimelineEventType.STOP.value:
#             carb.log_info("STOP event detected")

#         elif event_type == omni.timeline.TimelineEventType.PAUSE.value:
#             carb.log_info("PAUSE event detected")

#     timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(on_timeline_event)
#     carb.log_info("Timeline callback installed.")


# async def apply_initial_pose_async():
#     global _is_applying_pose

#     if _is_applying_pose:
#         return
#     _is_applying_pose = True

#     try:
#         # Timeline-Zustandswechsel wird erst im nächsten Frame wirksam
#         await omni.kit.app.get_app().next_update_async()
#         await omni.kit.app.get_app().next_update_async()

#         set_dodo_base_position()
#         set_dodo_joint_positions()

#         await omni.kit.app.get_app().next_update_async()
#         carb.log_info("Initial pose reapplied after PLAY.")
#     except Exception as e:
#         carb.log_error(f"Failed to apply initial pose: {e}")
#     finally:
#         _is_applying_pose = False


# def set_dodo_base_position():
#     dodo_xform = XFormPrim(prim_paths_expr=ROBOT_PRIM)
#     dodo_xform.set_world_poses(BASE_POS_INIT, BASE_ORI_INIT)


# def set_dodo_joint_positions():
#     robot = Articulation(prim_paths_expr=ROBOT_PRIM)
#     robot.initialize()

#     joint_names = robot.dof_names
#     carb.log_info(f"Articulation DOFs: {joint_names}")

#     q = np.zeros((1, len(joint_names)), dtype=np.float32)
#     for i, name in enumerate(joint_names):
#         if name in INIT_Q:
#             q[0, i] = INIT_Q[name]
#         else:
#             carb.log_warn(f"No initial value provided for joint '{name}', using 0.0")

#     qd = np.zeros_like(q)

#     robot.set_joint_positions(q)
#     robot.set_joint_velocities(qd)
#     robot.set_joints_default_state(positions=q, velocities=qd)

#     carb.log_info(f"Initial joint pose applied: {q}")


# main()