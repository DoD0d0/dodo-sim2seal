import omni.usd
import numpy as np
from isaacsim.core.prims import XFormPrim

xform = XFormPrim(prim_paths_expr="/dodobot_v3")
xform.set_world_poses(
	np.array([[0.0, 0.0, 0.575]], dtype=np.float32),
	np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
	)

