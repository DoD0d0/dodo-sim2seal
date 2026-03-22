import numpy as np
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics

joint_positions = {
    "left_joint_1":  -0.0,
    "right_joint_1": 0.0,
    "left_joint_2":  0.7,
    "right_joint_2": 0.7,
    "left_joint_3": -1.3,
    "right_joint_3": -1.3,
    "left_joint_4":  0.6,
    "right_joint_4": 0.6,
}

stage = get_current_stage()

for joint_name, angle_rad in joint_positions.items():
    for prim in stage.Traverse():
        if prim.GetName() == joint_name:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.GetTargetPositionAttr().Set(float(np.degrees(angle_rad)))
            print(f"Set {joint_name}: {angle_rad:.3f} rad")

print("Done. Save the USD if this pose looks right.")

