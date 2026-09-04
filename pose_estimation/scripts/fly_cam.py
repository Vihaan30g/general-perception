# === Paste into Script Editor and run ONCE ===
# Spawns ONLY a plain, standard UsdGeom.Camera for manual navigation.
# No light, no annotators, no capture logic.

import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

cam_path = "/World/FlyCam"
if not stage.GetPrimAtPath(cam_path).IsValid():
    camera = UsdGeom.Camera.Define(stage, cam_path)
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))

    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(1.5, 0.0, 1.0))
    xform.AddOrientOp().Set(Gf.Quatf(0.0,0.0,0.0,0.0))

    print(f"Created camera at '{cam_path}'.")
else:
    print(f"Camera already exists at '{cam_path}', leaving its current pose alone.")

print(
    f"\nSwitch the Viewport's camera dropdown (top-left) to '{cam_path}', "
    "then hold Right Mouse Button + W/A/S/D to move, Q/E for down/up, "
    "mouse to look around."
)