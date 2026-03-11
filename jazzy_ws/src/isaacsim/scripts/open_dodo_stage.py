"""Open a Dodo USD stage in Isaac Sim, set initial robot height and joint positions."""

import argparse
import asyncio
import numpy as np
import carb
import omni.usd
import omni.client
import omni.kit.app
import omni.kit.async_engine
import omni.timeline
from isaacsim.core.prims import XFormPrim, Articulation


# Default for current deployment USD (can be overridden via --base-height)
DEFAULT_BASE_HEIGHT = 0.575
BASE_ORI = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

# Training uses all-zero default joint positions
INIT_Q = {
    "left_joint_1": 0.0,
    "right_joint_1": 0.0,
    "left_joint_2": 0.0,
    "right_joint_2": 0.0,
    "left_joint_3": 0.0,
    "right_joint_3": 0.0,
    "left_joint_4": 0.0,
    "right_joint_4": 0.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to USD stage")
    parser.add_argument("--robot-prim", type=str, default="/dodobot_v3",
                        help="Prim path of the robot articulation")
    parser.add_argument("--base-height", type=float, default=DEFAULT_BASE_HEIGHT,
                        help="Initial root/base height in meters")
    parser.add_argument("--start-on-play", action="store_true",
                        help="Keep simulation running after loading")
    options = parser.parse_args()

    omni.kit.async_engine.run_coroutine(
        open_stage_async(
            options.path,
            options.robot_prim,
            options.start_on_play,
            options.base_height,
        )
    )


async def open_stage_async(path: str, robot_prim: str, start_on_play: bool, base_height: float):
    timeline = omni.timeline.get_timeline_interface()

    # Check that the file exists
    result, _ = await omni.client.stat_async(path)
    if result != omni.client.Result.OK:
        carb.log_error(f"USD file not found: {path}")
        return

    success, error = await omni.usd.get_context().open_stage_async(path)
    if not success:
        carb.log_error(f"Failed to open stage {path}: {error}")
        return

    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    # Set base position before play
    base_pos = np.array([[0.0, 0.0, base_height]], dtype=np.float32)
    xform = XFormPrim(prim_paths_expr=robot_prim)
    xform.set_world_poses(base_pos, BASE_ORI)

    # Play timeline to initialize physics
    timeline.play()
    await omni.kit.app.get_app().next_update_async()

    # Set initial joint positions
    robot = Articulation(prim_paths_expr=robot_prim)
    robot.initialize()

    dof_names = robot.dof_names
    carb.log_info(f"Articulation DOFs: {dof_names}")

    q = np.zeros((1, len(dof_names)), dtype=np.float32)
    for i, name in enumerate(dof_names):
        if name in INIT_Q:
            q[0, i] = INIT_Q[name]
        else:
            carb.log_warn(f"No init value for joint '{name}', using 0.0")

    qd = np.zeros_like(q)
    robot.set_joint_positions(q)
    robot.set_joint_velocities(qd)
    robot.set_joints_default_state(positions=q, velocities=qd)
    carb.log_info(f"Initial joint pose applied: {q}")

    await omni.kit.app.get_app().next_update_async()

    if not start_on_play:
        timeline.pause()


main()
