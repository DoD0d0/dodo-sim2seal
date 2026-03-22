import carb
import omni.usd
import numpy as np
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom

ROOT_ASSET_SCOPE_PATHS = (
    "/visuals",
    "/colliders",
    "/meshes",
)


def _hide_root_asset_scopes():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        carb.log_warn("No USD stage available to hide root asset scopes.")
        return

    for prim_path in ROOT_ASSET_SCOPE_PATHS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            carb.log_warn(f"Root asset scope is not imageable: {prim_path}")
            continue

        imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


xform = XFormPrim(prim_paths_expr="/dodobot_v3")
xform.set_world_poses(
    np.array([[0.0, 0.0, 0.575]], dtype=np.float32),
    np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
)
_hide_root_asset_scopes()
