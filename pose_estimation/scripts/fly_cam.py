# === Paste into Script Editor and RUN EVERY TIME you want to capture a view ===
# Fully self-contained -- doesn't rely on the Script Editor keeping variables
# alive between separate runs. Re-creating the camera/sensor wrapper each
# time is cheap and safe (it wraps the SAME underlying USD prim, it doesn't
# duplicate it). The view counter is persisted to a small file on disk so it
# survives even if you close and reopen the Script Editor.

import numpy as np
import os, json, cv2
from pxr import UsdGeom
import omni.usd
from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor

OUT_DIR = "/workspace/projects/general-perception/pose_estimation/data/captures_manual"
os.makedirs(OUT_DIR, exist_ok=True)

CAM_PRIM_PATH = "/World/Camera"
RESOLUTION_HW = (480, 640)   # (height, width)

stage = omni.usd.get_context().get_stage()
existing_prim = stage.GetPrimAtPath(CAM_PRIM_PATH)

# --- Create the camera prim ONLY if it doesn't already exist. If you've
# already run this once and moved the camera around in the viewport since,
# this will NOT reset its pose -- it just re-wraps the same prim. ---
cam = RtxCamera(CAM_PRIM_PATH)

if not existing_prim.IsValid():
    # first time ever: give it sane starting pose/lens settings
    cam.camera.set_focal_lengths([24.0])
    cam.camera.set_clipping_ranges([[0.01, 1000.0]])
    cam.set_world_poses(
        positions=np.array([[0.6, 0.0, 0.3]]),
        orientations=np.array([[0.5, -0.5, 0.5, -0.5]]),
    )
    print(f"Created new camera at '{CAM_PRIM_PATH}'. Switch the Viewport's "
          f"active camera to it, fly to a pose, then re-run this cell.")
else:
    # camera already exists (from a previous run / GUI creation) -- just use it
    sensor = CameraSensor(
        cam,
        resolution=RESOLUTION_HW,
        annotators=["rgb", "distance_to_image_plane"],
    )

    usd_camera = UsdGeom.Camera(existing_prim)

    def to_numpy(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    focal_length = usd_camera.GetFocalLengthAttr().Get()
    h_ap = usd_camera.GetHorizontalApertureAttr().Get()
    v_ap = usd_camera.GetVerticalApertureAttr().Get()
    height, width = RESOLUTION_HW
    fx = width * focal_length / h_ap
    fy = height * focal_length / v_ap
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    rgb_data, _ = sensor.get_data("rgb")
    depth_data, _ = sensor.get_data("distance_to_image_plane")
    rgb = to_numpy(rgb_data)
    depth = to_numpy(depth_data)

    positions, orientations = cam.get_world_poses()
    cam_pos = to_numpy(positions)[0].tolist()
    cam_quat_wxyz = to_numpy(orientations)[0].tolist()

    # --- persisted counter, survives across separate script runs ---
    counter_file = os.path.join(OUT_DIR, "_counter.txt")
    view_counter = int(open(counter_file).read().strip()) if os.path.exists(counter_file) else 0

    view_dir = os.path.join(OUT_DIR, f"view_{view_counter:02d}")
    os.makedirs(view_dir, exist_ok=True)

    depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(os.path.join(view_dir, "depth.npy"), depth_clean.astype(np.float32))

    cv2.imwrite(os.path.join(view_dir, "rgb.png"),
                cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))

    with open(os.path.join(view_dir, "intrinsics.json"), "w") as f:
        json.dump({"K": K.tolist(), "width": width, "height": height}, f)

    with open(os.path.join(view_dir, "camera_pose_debug.json"), "w") as f:
        json.dump({"position": cam_pos, "orientation_wxyz": cam_quat_wxyz}, f)

    with open(counter_file, "w") as f:
        f.write(str(view_counter + 1))

    print(f"Captured {view_dir}  (camera at {cam_pos})")