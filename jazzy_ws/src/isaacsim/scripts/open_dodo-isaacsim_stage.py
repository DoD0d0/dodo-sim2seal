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

"""Script to open a USD stage in Isaac Sim (in standard gui mode). This script along with its arguments are automatically passed to Isaac Sim via the ROS2 launch workflow"""

import carb
import argparse
import omni.usd
import asyncio
import omni.client
import omni.kit.async_engine
import omni.timeline

# imports for controlling the stage and prims after opening the stage (usd)
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.prims import Articulation

import numpy as np

ROBOT_PRIM = "/dodobot_v3" # change this to the prim path of your robot in the usd stage
BASE_POS_INIT = np.array([0, 0, 0.58]) # change this to the initial position of your robot base in the stage
BASE_QUAT_INIT = np.array([1, 0, 0, 0]) # change this to the initial orientation of your robot base in the stage (in quaternion format [w, x, y, z])
INIT_Q = {
        "left_joint_1": 0.0,
        "right_joint_1": 0.0,
        "left_joint_2": 0.4,
        "right_joint_2": 0.4,
        "left_joint_3": -0.7,
        "right_joint_3": -0.7,
        "left_joint_4": 0.3,
        "right_joint_4": 0.3,
    } # this is in radiands but we need degrees

#INIT_Q = {k: np.deg2rad(v) for k, v in INIT_Q.items()} # convert initial joint positions to radians

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
    if start_on_play:
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
            if timeline_interface is not None:
                await omni.kit.app.get_app().next_update_async()
                await omni.kit.app.get_app().next_update_async()
                # Set base pose via USD BEFORE physics starts
                await set_dodo_base_pose_usd_async()
                
                # set joints BEFORE physics starts
                await set_dodo_joints_async()

                # Start physics
                timeline_interface.play()

                # Wait a few frames for physics engine to initialize
                for _ in range(5):
                    await omni.kit.app.get_app().next_update_async()

                # Now set joints (physics is running, Articulation can initialize)
                await set_dodo_joints_async()

                # set_dodo_initial_pose() # set the initial pose of the robot after opening the stage
                # carb.log_info("Stage loaded and simulation is playing.")
                
                # # await omni.kit.app.get_app().next_update_async()
                # # await omni.kit.app.get_app().next_update_async()

                # # set base pose in USD BEFORE physics starts
                # await set_dodo_base_pose_usd_async()

                # # set joints BEFORE physics starts
                # await set_dodo_joints_async()

                # # now start physics
                # timeline_interface.play()
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

def set_root_pose_usd(stage, prim_path: str, pos, quat_wxyz):
    from pxr import UsdGeom, Gf

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim at {prim_path}")

    xform = UsdGeom.Xformable(prim)

    # find or create ops
    translate_op = None
    orient_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if orient_op is None:
        orient_op = xform.AddOrientOp()

    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    orient_op.Set(Gf.Quatd(float(quat_wxyz[0]),
                           Gf.Vec3d(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))))
    
async def set_dodo_base_pose_usd_async():
    from pxr import UsdPhysics

    stage = get_current_stage()

    # find articulation root under ROBOT_PRIM
    root_path = ROBOT_PRIM
    for p in stage.Traverse():
        if p.GetPath().pathString.startswith(ROBOT_PRIM):
            if UsdPhysics.ArticulationRootAPI(p).GetPrim().IsValid():
                root_path = p.GetPath().pathString
                carb.log_info(f"Using articulation root: {root_path}")
                break

    set_root_pose_usd(stage, root_path, BASE_POS_INIT, BASE_QUAT_INIT)
    carb.log_info("Set base pose via USD Xform.")

async def set_dodo_joints_async():
    from pxr import UsdPhysics

    stage = get_current_stage()

    root_path = ROBOT_PRIM
    for p in stage.Traverse():
        if p.GetPath().pathString.startswith(ROBOT_PRIM):
            if UsdPhysics.ArticulationRootAPI(p).GetPrim().IsValid():
                root_path = p.GetPath().pathString
                break

    art = Articulation(root_path)

    # retry initialize until physics view exists
    for _ in range(60):
        try:
            art.initialize()
            break
        except AttributeError:
            await omni.kit.app.get_app().next_update_async()
    else:
        raise RuntimeError("Physics not ready: Articulation.initialize() keeps failing")
    
    carb.log_info(f"[ART] root_path: {root_path}")
    carb.log_info(f"[ART] dof_names ({len(art.dof_names)}): {art.dof_names}")
    q = art.get_joint_positions()
    carb.log_info(f"[ART] q shape: {np.array(q).shape}  q: {q}")

    joint_names = art.dof_names
    #q = art.get_joint_positions()
    carb.log_info(f"DOF names: {joint_names}")
    carb.log_info(f"q before: {q}")

    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    for jn, val in INIT_Q.items():
        if jn in name_to_idx:
            q[name_to_idx[jn]] = float(val)
        else:
            carb.log_warn(f"[WARN] Joint name {jn} not found in articulation DOFs.")

    art.set_joint_positions(q)
    art.set_joint_velocities(np.zeros_like(q))
    carb.log_info("Set joint positions via Articulation.")

    
main()
