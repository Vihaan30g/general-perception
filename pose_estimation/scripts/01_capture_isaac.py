# # scripts/01_capture_isaac.py
# from isaacsim import SimulationApp
# simulation_app = SimulationApp({"headless": True})   # set False to watch it capture

# import os, json
# import numpy as np
# import cv2
# from scipy.spatial.transform import Rotation as R
# from pxr import UsdGeom

# from isaacsim.core.api import World
# from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
# from isaacsim.core.prims import SingleXFormPrim
# from isaacsim.storage.native import get_assets_root_path
# from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
# import isaacsim.core.experimental.utils.app as app_utils

# # container path (matches your mounted volume: host /opt/isaac-sim/projects
# # <-> container /workspace/projects)


# print("\n\nBREAK0\n\n")



# OUT_DIR = "/workspace/projects/general-perception/pose_estimation/data/captures"
# os.makedirs(OUT_DIR, exist_ok=True)

# RESOLUTION_HW = (480, 640)               # (height, width)

# world = World(stage_units_in_meters=1.0)
# world.scene.add_default_ground_plane()

# # --- Add object ---
# assets_root = get_assets_root_path()
# object_usd_path = assets_root + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"
# add_reference_to_stage(usd_path=object_usd_path, prim_path="/World/Object")
# object_center = np.array([0.0, 0.0, 0.1])   # roughly where the object sits

# # --- Add camera (new RtxCamera / CameraSensor API) ---
# # NOTE: aux_output_level was removed here -- your installed build's
# # RtxCamera.__init__ doesn't accept it. Run scripts/00_inspect_api.py if you
# # hit further TypeErrors on other kwargs (schemas, tick_rate, etc.) and swap
# # in whatever the real signature reports.
# CAM_PRIM_PATH = "/World/Camera"
# cam = RtxCamera(CAM_PRIM_PATH, tick_rate=0)

# # Focal length / clipping range setter names may also differ -- guarded so
# # the script keeps going and tells you exactly what's wrong instead of
# # dying silently.
# try:
#     cam.camera.set_focal_lengths(24.0)
# except AttributeError as e:
#     print(f"[WARN] set_focal_lengths not available on cam.camera ({e}); "
#           f"leaving default focal length. Check 00_inspect_api.py output "
#           f"for the correct method name.")

# try:
#     cam.camera.set_clipping_ranges(0.01, 1000.0)
# except AttributeError as e:
#     print(f"[WARN] set_clipping_ranges not available on cam.camera ({e}); "
#           f"leaving default clipping range.")

# sensor = CameraSensor(
#     cam,
#     resolution=RESOLUTION_HW,
#     annotators=["rgb", "distance_to_image_plane"],
# )

# cam_xform = SingleXFormPrim(CAM_PRIM_PATH)   # used to move the camera each view
# world.reset()
# app_utils.play(commit=True)

# # --- Read intrinsics-relevant attributes directly from the USD camera prim ---
# usd_camera = UsdGeom.Camera(get_current_stage().GetPrimAtPath(CAM_PRIM_PATH))
# focal_length = usd_camera.GetFocalLengthAttr().Get()
# h_aperture = usd_camera.GetHorizontalApertureAttr().Get()
# v_aperture = usd_camera.GetVerticalApertureAttr().Get()
# height, width = RESOLUTION_HW
# fx = width * focal_length / h_aperture
# fy = height * focal_length / v_aperture
# cx, cy = width / 2.0, height / 2.0
# K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])



# print("\n\nBREAK1\n\n")




# # --- Helper: look-at quaternion (Isaac uses wxyz) ---
# def look_at_quat(eye, target, up=np.array([0.0, 0.0, 1.0])):
#     eye, target = np.array(eye, dtype=float), np.array(target, dtype=float)
#     forward = target - eye
#     forward /= np.linalg.norm(forward)
#     right = np.cross(forward, up)
#     right /= np.linalg.norm(right)
#     true_up = np.cross(right, forward)
#     rot_mat = np.stack([right, true_up, -forward], axis=1)   # camera looks down local -Z
#     quat_xyzw = R.from_matrix(rot_mat).as_quat()
#     return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # wxyz

# # --- Orbit sampling: 2 elevation rings x N azimuths ---
# def orbit_positions(center, radius, n_azimuth, elevations_deg):
#     poses = []
#     for elev_deg in elevations_deg:
#         elev = np.radians(elev_deg)
#         for i in range(n_azimuth):
#             az = 2 * np.pi * i / n_azimuth
#             x = center[0] + radius * np.cos(elev) * np.cos(az)
#             y = center[1] + radius * np.cos(elev) * np.sin(az)
#             z = center[2] + radius * np.sin(elev)
#             poses.append(np.array([x, y, z]))
#     return poses

# positions = orbit_positions(object_center, radius=0.6, n_azimuth=10, elevations_deg=[15, 45])





# print("\n\nBREAK2\n\n")



# # --- Capture loop ---
# for idx, pos in enumerate(positions):
#     quat = look_at_quat(pos, object_center)
#     cam_xform.set_world_pose(position=pos, orientation=quat)

#     for _ in range(5):        # step a few frames so the render settles
#         world.step(render=True)

#     rgb_data, _ = sensor.get_data("rgb")                          # (H, W, 3 or 4) uint8
#     depth_data, _ = sensor.get_data("distance_to_image_plane")    # (H, W) float32 meters

#     rgb = np.asarray(rgb_data)
#     depth = np.asarray(depth_data)

#     view_dir = os.path.join(OUT_DIR, f"view_{idx:02d}")
#     os.makedirs(view_dir, exist_ok=True)

#     depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
#     np.save(os.path.join(view_dir, "depth.npy"), depth_clean.astype(np.float32))

#     cv2.imwrite(os.path.join(view_dir, "rgb.png"),
#                 cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))

#     with open(os.path.join(view_dir, "intrinsics.json"), "w") as f:
#         json.dump({"K": K.tolist(), "width": width, "height": height}, f)

#     print(f"Saved {view_dir}")

# simulation_app.close()




# print("\n\nBREAK3\n\n")










# scripts/01_capture_isaac.py
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})   # set False to watch it capture

import os, json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pxr import UsdGeom

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
import isaacsim.core.experimental.utils.app as app_utils

# container path (matches your mounted volume: host /opt/isaac-sim/projects
# <-> container /workspace/projects)
OUT_DIR = "/workspace/projects/general-perception/pose_estimation/data/captures"
os.makedirs(OUT_DIR, exist_ok=True)

RESOLUTION_HW = (480, 640)               # (height, width)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# --- Add object ---
assets_root = get_assets_root_path()
object_usd_path = assets_root + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"
add_reference_to_stage(usd_path=object_usd_path, prim_path="/World/Object")
object_center = np.array([0.0, 0.0, 0.1])   # roughly where the object sits

# --- Add camera (confirmed signature: no aux_output_level, tick_rate is fine) ---
CAM_PRIM_PATH = "/World/Camera"
cam = RtxCamera(CAM_PRIM_PATH, tick_rate=0)

cam.camera.set_focal_lengths([24.0])            # array-form API -> pass as a list
cam.camera.set_clipping_ranges([[0.01, 1000.0]])

sensor = CameraSensor(
    cam,
    resolution=RESOLUTION_HW,
    annotators=["rgb", "distance_to_image_plane"],
)

world.reset()
app_utils.play(commit=True)

# --- Read intrinsics-relevant attributes directly from the USD camera prim ---
usd_camera = UsdGeom.Camera(get_current_stage().GetPrimAtPath(CAM_PRIM_PATH))
focal_length = usd_camera.GetFocalLengthAttr().Get()
h_aperture = usd_camera.GetHorizontalApertureAttr().Get()
v_aperture = usd_camera.GetVerticalApertureAttr().Get()
height, width = RESOLUTION_HW
fx = width * focal_length / h_aperture
fy = height * focal_length / v_aperture
cx, cy = width / 2.0, height / 2.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# --- Helper: look-at quaternion (Isaac uses wxyz) ---
def look_at_quat(eye, target, up=np.array([0.0, 0.0, 1.0])):
    eye, target = np.array(eye, dtype=float), np.array(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rot_mat = np.stack([right, true_up, -forward], axis=1)   # camera looks down local -Z
    quat_xyzw = R.from_matrix(rot_mat).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # wxyz

# --- Orbit sampling: 2 elevation rings x N azimuths ---
def orbit_positions(center, radius, n_azimuth, elevations_deg):
    poses = []
    for elev_deg in elevations_deg:
        elev = np.radians(elev_deg)
        for i in range(n_azimuth):
            az = 2 * np.pi * i / n_azimuth
            x = center[0] + radius * np.cos(elev) * np.cos(az)
            y = center[1] + radius * np.cos(elev) * np.sin(az)
            z = center[2] + radius * np.sin(elev)
            poses.append(np.array([x, y, z]))
    return poses

positions = orbit_positions(object_center, radius=0.6, n_azimuth=10, elevations_deg=[15, 45])

# get_data() returns warp arrays in this build (even though docs say
# "numpy/warp"). np.asarray() on a warp array hits an indexing bug here --
# always go through .numpy() explicitly when the object has it.
def to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)

# --- Capture loop ---
for idx, pos in enumerate(positions):
    quat = look_at_quat(pos, object_center)

    # Array-form API: pass as (1, 3) / (1, 4) arrays even for a single camera
    cam.set_world_poses(
        positions=np.array([pos]),
        orientations=np.array([quat]),
    )

    for _ in range(5):        # step a few frames so the render settles
        world.step(render=True)

    rgb_data, _ = sensor.get_data("rgb")                          # (H, W, 3 or 4)
    depth_data, _ = sensor.get_data("distance_to_image_plane")    # (H, W)

    rgb = to_numpy(rgb_data)
    depth = to_numpy(depth_data)

    view_dir = os.path.join(OUT_DIR, f"view_{idx:02d}")
    os.makedirs(view_dir, exist_ok=True)

    depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(os.path.join(view_dir, "depth.npy"), depth_clean.astype(np.float32))

    cv2.imwrite(os.path.join(view_dir, "rgb.png"),
                cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))

    with open(os.path.join(view_dir, "intrinsics.json"), "w") as f:
        json.dump({"K": K.tolist(), "width": width, "height": height}, f)

    print(f"Saved {view_dir}")

simulation_app.close()