from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics

ROBOT_PRIM = "/dodobot_v3"
STIFFNESS = 60.0
DAMPING = 3.0

stage = get_current_stage()

for prim in stage.Traverse():
    path = prim.GetPath().pathString
    name = prim.GetName()
    if not path.startswith(ROBOT_PRIM):
        continue
    if not name.startswith(("left_joint_", "right_joint_")):
        continue

    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")

    drive.GetStiffnessAttr().Set(STIFFNESS)
    drive.GetDampingAttr().Set(DAMPING)
    print(f"{name}: stiffness={STIFFNESS}, damping={DAMPING}")

print("Done.")

