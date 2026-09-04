# 6D Pose Estimation Mini-Project: Isaac Sim + Open3D

Full implementation guide. No theory — only what to run and why.

---

## 0. Project Structure

```
project/
├── data/
│   └── captures/
│       ├── view_00/
│       │   ├── rgb.png
│       │   ├── depth.npy          # float32 meters, HxW
│       │   └── intrinsics.json
│       ├── view_01/
│       └── ...
├── scripts/
│   ├── 01_capture_isaac.py        # runs INSIDE Isaac Sim python
│   ├── 02_pcd_from_depth.py
│   ├── 03_registration.py
│   ├── 04_plane_and_object_segmentation.py
│   ├── 05_pose_estimation_pca.py
│   └── 06_visualize.py
└── output/
    ├── merged_scene.pcd
    ├── object.pcd
    └── pose.json
```

Environment:
```bash
# Open3D pipeline runs in YOUR OWN venv (not Isaac's bundled python):
python3 -m venv .venv && source .venv/bin/activate
pip install open3d numpy scipy opencv-python   # any recent opencv-python (>=4.5) is fine —
                                                 # it's only used for image read/write here

# Isaac Sim capture script runs with Isaac Sim's own python, INSIDE the container:
./python.sh /PATH/scripts/01_capture_isaac.py
```

### 0.1 Docker specifics (Isaac Sim 6.0.1 container)

Since Isaac Sim runs in a container and only your mounted directory persists to disk, put **both** the capture script and the output folder under that mount so they're visible from the container and from your host venv:

```
/PATH/                      <- replace PATH with your actual host-mounted dir
├── scripts/
│   └── 01_capture_isaac.py
└── data/captures/          <- written by the container, read later by your host venv
```

Run the capture step from a shell inside the running container:
```bash
docker exec -it <isaac_sim_container_name> bash
cd /isaac-sim   # or wherever ./python.sh lives in the image
./python.sh /PATH/scripts/01_capture_isaac.py
```

Everything after Step 1 (Steps 2–6) runs entirely in your host venv against files under `/PATH/data/captures` — no need to touch the container again once capture is done.

---

## 1. Capturing Depth from Isaac Sim

### 1.1 Which camera to use

**Important for 6.0.1 specifically:** the `isaacsim.sensors.camera.Camera` class (which most older tutorials use) is **deprecated as of Isaac Sim 6.0**. It still loads, but the supported path now is `isaacsim.sensors.experimental.rtx`, split into:
- `RtxCamera` — authoring: creates/wraps the USD camera prim, sets focal length / clipping range.
- `CameraSensor` — runtime: wraps an `RtxCamera`, attaches Replicator annotators, and gives you `get_data("annotator-name")`.

Depth annotator to attach: **`distance_to_image_plane`** (NOT `distance_to_camera`). It's the z-depth along the camera's optical axis, which is exactly what the pinhole back-projection formula in Step 2 assumes. `distance_to_camera` is radial distance from the camera origin and would need a different formula.

Camera convention confirmed by NVIDIA's own docs: it looks down local **-Z**, with **+Y up** and **+X right** — same as I assumed earlier, so the `look_at_quat` helper below is unchanged.

### 1.2 Scene setup + orbit capture script

```python
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
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
import isaacsim.core.experimental.utils.app as app_utils

OUT_DIR = "/PATH/data/captures"          # <-- replace /PATH with your mounted dir
os.makedirs(OUT_DIR, exist_ok=True)

RESOLUTION_HW = (480, 640)               # (height, width) -- note the order

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# --- Add object ---
assets_root = get_assets_root_path()
object_usd_path = assets_root + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"
add_reference_to_stage(usd_path=object_usd_path, prim_path="/World/Object")
object_center = np.array([0.0, 0.0, 0.1])   # roughly where the object sits

# --- Add camera (new RtxCamera / CameraSensor API) ---
CAM_PRIM_PATH = "/World/Camera"
cam = RtxCamera(CAM_PRIM_PATH, tick_rate=0, aux_output_level="NONE")
cam.camera.set_focal_lengths(24.0)          # mm-equivalent in Kit units
cam.camera.set_clipping_ranges(0.01, 1000.0)

sensor = CameraSensor(
    cam,
    resolution=RESOLUTION_HW,
    annotators=["rgb", "distance_to_image_plane"],
)

cam_xform = SingleXFormPrim(CAM_PRIM_PATH)   # used to move the camera each view
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

# --- Capture loop ---
for idx, pos in enumerate(positions):
    quat = look_at_quat(pos, object_center)
    cam_xform.set_world_pose(position=pos, orientation=quat)

    for _ in range(5):        # step a few frames so the render settles
        world.step(render=True)

    rgb_data, _ = sensor.get_data("rgb")                          # (H, W, 3 or 4) uint8
    depth_data, _ = sensor.get_data("distance_to_image_plane")    # (H, W) float32 meters

    rgb = np.asarray(rgb_data)
    depth = np.asarray(depth_data)

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
```

Notes:
- `resolution` for `CameraSensor` is `(height, width)` — easy to get backwards, double-check your saved images aren't transposed.
- Reading intrinsics straight off the `UsdGeom.Camera` prim attributes (as above) is the most version-proof approach — it doesn't depend on whichever convenience method does or doesn't exist on the sensor wrapper in a given release.
- Depth on disk: `.npy`, `float32`, values in **meters**, `0` = invalid/no-hit. This is the simplest lossless format. If you want PNG-compatible depth (like standard RGBD datasets), instead save `(depth * 1000).astype(np.uint16)` — a 16-bit PNG in millimeters, writable with `cv2.imwrite(path, depth_mm)`.
- If `RtxCamera`/`CameraSensor` aren't importable in your exact build, fall back to the deprecated `isaacsim.sensors.camera.Camera` class — it still works in 6.0.1, just prints a deprecation warning; the `get_rgba()` / `get_depth()` methods from the original version of this guide apply directly in that case.

---

## 2. Depth → Point Cloud (per view)

Standard pinhole back-projection (matches `distance_to_image_plane`):

```python
# scripts/02_pcd_from_depth.py
import numpy as np
import open3d as o3d
import json, os

def load_view(view_dir):
    depth = np.load(os.path.join(view_dir, "depth.npy"))
    rgb = o3d.io.read_image(os.path.join(view_dir, "rgb.png"))
    with open(os.path.join(view_dir, "intrinsics.json")) as f:
        meta = json.load(f)
    K = np.array(meta["K"])
    return depth, np.asarray(rgb), K, meta["width"], meta["height"]

def depth_to_pointcloud(depth, K, rgb=None, depth_trunc=3.0, depth_min=0.05):
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid = (z > depth_min) & (z < depth_trunc)

    x = (i - cx) * z / fx
    y = (j - cy) * z / fy      # OpenCV/pinhole convention: x-right, y-down, z-forward

    pts = np.stack([x, y, z], axis=-1)[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if rgb is not None:
        colors = rgb[valid].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

if __name__ == "__main__":
    view_dirs = sorted(
        os.path.join("data/captures", d) for d in os.listdir("data/captures")
    )
    os.makedirs("output/per_view_pcd", exist_ok=True)

    for view_dir in view_dirs:
        depth, rgb, K, w, h = load_view(view_dir)
        pcd = depth_to_pointcloud(depth, K, rgb)
        name = os.path.basename(view_dir)
        o3d.io.write_point_cloud(f"output/per_view_pcd/{name}.pcd", pcd)
        print(f"{name}: {len(pcd.points)} points")
```

Each resulting cloud is in **that camera's own local frame** (z-forward, y-down, x-right — this is NOT the same as Isaac Sim's world/camera-prim convention, which is y-up/z-backward at the prim level; the back-projection above already gives you the standard vision convention, which is what Open3D/OpenCV expect).

---

## 3. Multi-View Registration (merging into one full point cloud)

We treat the per-view clouds as **unregistered** (no ground-truth extrinsics used) — this is the actual point-cloud-registration exercise.

Pipeline: **downsample → normals → FPFH features → RANSAC global registration (coarse) → point-to-plane ICP (fine) → pose-graph multiway optimization → merge.**

```python
# scripts/03_registration.py
import numpy as np
import open3d as o3d
import glob

VOXEL = 0.005   # 5mm — tune based on object scale

def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh

def global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size):
    dist_thresh = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        dist_thresh,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )
    return result

def refine_icp(src, tgt, voxel_size, init_transform):
    dist_thresh = voxel_size * 0.4
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, dist_thresh, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def pairwise_registration(src, tgt, voxel_size):
    src_down, src_fpfh = preprocess(src, voxel_size)
    tgt_down, tgt_fpfh = preprocess(tgt, voxel_size)
    coarse = global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size)
    fine = refine_icp(src_down, tgt_down, voxel_size, coarse.transformation)
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src_down, tgt_down, voxel_size * 1.5, fine.transformation)
    return fine.transformation, information

def full_registration(pcds, voxel_size):
    n = len(pcds)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for src_id in range(n):
        for tgt_id in range(src_id + 1, n):
            transform, info = pairwise_registration(pcds[src_id], pcds[tgt_id], voxel_size)
            if tgt_id == src_id + 1:      # sequential (odometry) edge
                odometry = transform @ odometry
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src_id, tgt_id, transform, info, uncertain=False))
            else:                          # loop-closure edge
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src_id, tgt_id, transform, info, uncertain=True))
    return pose_graph

if __name__ == "__main__":
    files = sorted(glob.glob("output/per_view_pcd/*.pcd"))
    pcds = [o3d.io.read_point_cloud(f) for f in files]

    pose_graph = full_registration(pcds, VOXEL)

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=VOXEL * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    merged = o3d.geometry.PointCloud()
    for i, pcd in enumerate(pcds):
        pcd_t = pcd.transform(pose_graph.nodes[i].pose)
        merged += pcd_t

    merged = merged.voxel_down_sample(VOXEL)
    o3d.io.write_point_cloud("output/merged_scene.pcd", merged)
    print(f"Merged cloud: {len(merged.points)} points -> output/merged_scene.pcd")
```

Tuning notes:
- `VOXEL` should be ~1-2% of the object's bounding box size.
- If global registration keeps failing (few/no correct RANSAC correspondences), your object may lack geometric features from some angles — increase FPFH search radius or reduce voxel size.
- With only ~10-20 views this all-pairs loop is cheap. For many more views, only register neighboring views + a handful of loop-closure pairs instead of every pair.

---

## 4. Plane Segmentation + Object Isolation

```python
# scripts/04_plane_and_object_segmentation.py
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("output/merged_scene.pcd")

# --- 1. Statistical outlier removal (clean registration artifacts) ---
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# --- 2. Plane (table) segmentation via RANSAC ---
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.008, ransac_n=3, num_iterations=1000)
object_cloud = pcd.select_by_index(inliers, invert=True)   # everything NOT the plane

# --- 3. Radius outlier removal on the remainder ---
object_cloud, _ = object_cloud.remove_radius_outlier(nb_points=16, radius=0.02)

# --- 4. DBSCAN clustering to isolate just the object (drop stray noise clusters) ---
labels = np.array(object_cloud.cluster_dbscan(eps=0.015, min_points=20, print_progress=True))
if labels.max() < 0:
    raise RuntimeError("No clusters found — loosen eps/min_points")

largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
object_final = object_cloud.select_by_index(np.where(labels == largest_cluster)[0])

# --- 5. Final voxel downsample (optional, for speed downstream) ---
object_final = object_final.voxel_down_sample(voxel_size=0.003)

o3d.io.write_point_cloud("output/object.pcd", object_final)
print(f"Object isolated: {len(object_final.points)} points -> output/object.pcd")
```

---

## 5. 6D Pose Estimation via PCA

```python
# scripts/05_pose_estimation_pca.py
import numpy as np
import open3d as o3d
import json

pcd = o3d.io.read_point_cloud("output/object.pcd")
points = np.asarray(pcd.points)

# --- Translation: centroid ---
centroid = points.mean(axis=0)
centered = points - centroid

# --- Rotation: eigenvectors of the covariance matrix ---
cov = np.cov(centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)          # ascending eigenvalue order

order = np.argsort(eigvals)[::-1]               # descending: largest-spread axis first
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# Ensure a right-handed rotation matrix (determinant +1)
if np.linalg.det(eigvecs) < 0:
    eigvecs[:, -1] *= -1

R = eigvecs                                      # columns = object's principal axes in world frame

T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = centroid

print("Rotation:\n", R)
print("Translation:", centroid)

with open("output/pose.json", "w") as f:
    json.dump({"rotation": R.tolist(), "translation": centroid.tolist(),
               "transform_4x4": T.tolist()}, f, indent=2)
```

Known limitation to be aware of (not theory, just a practical caveat): PCA gives you the three principal axes but each axis's **sign/direction is ambiguous** (eigenvectors are defined up to sign), and for objects with near-equal extents along two axes the eigenvector ordering can become unstable/swap. This is fine for a first pass; if you want a more reliable pose later, refine with ICP against a known CAD/reference model of the object (register `object.pcd` to the model using the same `global_registration` + `refine_icp` functions from Step 3 — that gives you a much more robust and unambiguous 6D pose than PCA alone).

---

## 6. Visualization

```python
# scripts/06_visualize.py
import open3d as o3d
import numpy as np
import json

pcd = o3d.io.read_point_cloud("output/object.pcd")
with open("output/pose.json") as f:
    pose = json.load(f)
T = np.array(pose["transform_4x4"])

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
frame.transform(T)

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)  # origin reference

o3d.visualization.draw_geometries([pcd, frame, world_frame])
```

---

## 7. Suggested Build Order (checklist)

1. Run `01_capture_isaac.py` inside Isaac Sim; confirm each `view_XX/` folder has `rgb.png` + `depth.npy` and `depth.npy` isn't all zeros.
2. Run `02_pcd_from_depth.py`; open a couple of the resulting `.pcd` files individually in `o3d.visualization.draw_geometries` to sanity-check they look like partial object+table scans, not garbage (wrong axis signs show up immediately as inside-out geometry).
3. Run `03_registration.py`; visualize `merged_scene.pcd` — you should see one coherent, fairly complete point cloud of the object sitting on the table, with no obvious double-walls (double-walls = failed registration on some pair).
4. Run `04_plane_and_object_segmentation.py`; visualize `object.pcd` — should be just the object, no table remnants.
5. Run `05_pose_estimation_pca.py`.
6. Run `06_visualize.py` and check the coordinate frame is centered on the object, with axes roughly aligned to its edges.

Once this works end-to-end, natural extensions: swap PCA for ICP-to-CAD-model refinement, add more orbit rings for better top/bottom coverage, or try a learned method (e.g. a quick FoundationPose/6D-diffusion baseline) for comparison against your classical pipeline.
