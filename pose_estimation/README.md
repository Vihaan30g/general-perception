# Pose Estimation via Multi-View Point Cloud Reconstruction

> Part of the [`general-perception`](https://github.com/Vihaan30g/general-perception) repository — a collection of perception projects spanning 3D reconstruction, segmentation, and scene understanding.

---

## Overview

This project implements a complete **6-DoF object pose estimation pipeline** using synthetic data generated in **NVIDIA Isaac Sim**. The pipeline takes multi-view RGB-D captures of a scene, reconstructs a merged 3D point cloud using two independent registration methods, isolates the object of interest, and estimates its 6-DoF pose (position + orientation) using PCA on the resulting point cloud geometry.

The primary goal was hands-on familiarisation with **Open3D**, **Isaac Sim's Replicator**, and the full stack of 3D perception — from raw depth backprojection through to pose output — without relying on any learned models. Every stage is implemented from scratch using classical geometric methods.

**Capture data (6 frames) is included in the repository**, so the entire pipeline from Step 1 onwards can be reproduced without access to Isaac Sim.

---

## Results

### Isaac Sim Simulation Scene

The scene used for data capture — objects placed on a ground plane, lit and rendered in Isaac Sim with the Replicator BasicWriter writing out RGB, depth, and camera parameters for each viewpoint.

![Isaac Sim Scene](images/isaac_sim_scene.png)

---

### Single-Frame Point Cloud

Point cloud reconstructed from a single depth frame by backprojecting the depth map through the camera intrinsics into the camera-local coordinate frame. Each pixel with a valid depth value becomes a 3D point, coloured from the corresponding RGB image.

![Single Frame Point Cloud](images/single_frame_pcd.png)

---

### Merged Multi-View Point Cloud

All 6 per-view point clouds registered and merged into a single world-space point cloud. Two registration methods were implemented: **Method 2a** using Isaac's ground-truth camera poses directly, and **Method 2b** using FPFH feature matching + RANSAC global alignment + point-to-plane ICP + pose graph optimisation — no pose priors required.

![Merged Multi-View Point Cloud](images/merged_pcd.png)

---

### Isolated Object Cloud (After Plane Segmentation)

The ground plane is removed via RANSAC plane fitting. Statistical and radius outlier removal cleans registration artefacts. DBSCAN clustering extracts the largest remaining cluster — the object of interest — discarding stray noise points.

![Isolated Object Point Cloud](images/object_pcd.png)

---

### Final Pose Estimation Result

The estimated 6-DoF pose visualised as a coordinate frame (RGB axes = X/Y/Z) aligned to the object's principal axes, alongside the world-origin reference frame. Translation is the point cloud centroid; rotation is derived from the eigenvectors of the point covariance matrix (PCA).

![Final Pose Estimation](images/pose_result.png)

---

## Pipeline Overview

The pipeline is structured as a sequence of numbered scripts, each corresponding to one stage of the process.

```
Isaac Sim Replicator
    │  RGB images, depth maps, camera params (6 frames)
    ▼
[Step 0]  Verify camera conventions against Isaac's own ground-truth pointcloud
    │
    ▼
[Step 1]  Build per-view point clouds from depth + intrinsics (camera-local frame)
    │
    ├──[Step 2a]  Merge views using ground-truth extrinsics (known camera poses)
    │
    └──[Step 2b]  Merge views using FPFH + RANSAC + ICP registration (no pose priors)
    │
    ▼
[Step 3]  Segment and isolate the object (remove ground plane, denoise, cluster)
    │
    ▼
[Step 4]  Estimate 6-DoF pose via PCA on the isolated object cloud
    │
    ▼
[Step 5]  Visualize: object cloud + estimated pose frame + world origin
```

---

## Repository Structure

```
pose_estimation/
├── data/
│   └── captures/
│       ├── frame1/          # RGB image, depth map, camera params, GT pointcloud
│       ├── frame2/
│       ├── frame3/
│       ├── frame4/
│       ├── frame5/
│       └── frame6/
├── images/                  # Screenshots used in this README
│   ├── isaac_sim_scene.png
│   ├── single_frame_pcd.png
│   ├── merged_pcd.png
│   ├── object_pcd.png
│   └── pose_result.png
├── scenes/
│   └── objects_on_ground.usd   # Isaac Sim scene used for capture (requires Isaac Sim)
└── scripts/
    ├── utils_isaac_io.py           # Shared IO and camera math (read this first)
    ├── 00_verify_conventions.py    # Step 0: Sanity-check depth backprojection math
    ├── 01_pcd_from_depth.py        # Step 1: Build per-view point clouds
    ├── 02a_registration_known_extrinsics.py  # Step 2a: Merge via GT camera poses
    ├── 02b_registration_icp.py     # Step 2b: Merge via FPFH+RANSAC+ICP
    ├── 03_plane_and_object_segmentation.py   # Step 3: Isolate object
    ├── 04_pose_estimation_pca.py   # Step 4: Estimate 6-DoF pose
    ├── 05_visualize.py             # Step 5: Visualize result
    ├── fly_cam.py                  # Isaac Sim utility: add a free-fly camera
    └── view_pointcloud.py          # Utility: inspect any .pcd file at any stage
```

> **Note:** `outputs/` is not committed to the repository (it would contain large `.pcd` files). It is created automatically when you run the scripts.

---

## Technical Details

### Camera Convention

Isaac Sim uses a **USD / Omniverse coordinate system** that requires careful handling:

- World frame: right-handed, Y-up.
- Camera-local frame: OpenGL-style — X-right, Y-up, camera looks down **−Z**.
- `cameraViewTransform` in the JSON is a **row-major, row-vector view matrix**, meaning `p_world_row @ M = p_camera_row`. To use it as a standard column-vector transform, it must be transposed and inverted.
- `distance_to_image_plane` is **z-depth** (axial distance along the optical axis), not Euclidean ray length.

These conventions are documented in detail in `utils_isaac_io.py` and verified automatically by `00_verify_conventions.py`.

### Intrinsics: Why Not Aperture + Focal Length?

Intrinsics are derived from Isaac's `cameraProjection` matrix rather than the physical `cameraAperture` and `cameraFocalLength` parameters. Isaac's "aperture fit" conform policy can silently rescale the effective FOV on one axis without updating the reported aperture values. On real capture data, this caused a ~23% systematic error in `fy` when derived from aperture — deriving from the projection matrix used by the renderer eliminates this ambiguity entirely.

### Registration: Two Methods

**Method 2a — Known Extrinsics:** Uses Isaac's ground-truth `cameraViewTransform` to directly place each view into world space. No feature matching needed — just transform and concatenate. This serves as the reference for validating Method 2b.

**Method 2b — ICP:** Registers views purely from geometry, without any pose priors. Per-view clouds are preprocessed (voxel downsample + normal estimation + FPFH features), then pairwise registered using coarse RANSAC global alignment followed by point-to-plane ICP refinement. A pose graph is constructed with odometry edges (consecutive pairs) and loop-closure edges (all non-consecutive pairs), then globally optimised with Levenberg-Marquardt.

### Pose Estimation

Pose is estimated using **PCA on the isolated object point cloud**:

- **Translation:** centroid of the point cloud.
- **Rotation:** eigenvectors of the point covariance matrix, sorted by descending eigenvalue (largest spread = primary axis). The rotation matrix is corrected to right-handedness (det = +1) if the eigenvectors form a left-handed system.

This approach is geometry-only and makes no assumptions about object category or shape — it estimates the principal axes of the observed geometry.

---

## Setup

### Prerequisites

- Python 3.8+
- Open3D
- NumPy

Isaac Sim is **not required** to run the pipeline. The 6-frame capture data is included in `data/captures/`.

### Installation

```bash
# Clone only the pose_estimation subfolder using sparse checkout
git clone --filter=blob:none --sparse https://github.com/Vihaan30g/general-perception.git
cd general-perception
git sparse-checkout set pose_estimation

# Install dependencies
pip install open3d numpy
```

> Alternatively, clone the full repository if you plan to explore other projects in it:
> ```bash
> git clone https://github.com/Vihaan30g/general-perception.git
> cd general-perception/pose_estimation
> pip install open3d numpy
> ```

All scripts must be run from the `pose_estimation/scripts/` directory so that relative paths to `../data/` and `../outputs/` resolve correctly.

```bash
cd pose_estimation/scripts
```

---

## Running the Pipeline

### Step 0 — Verify Camera Conventions *(run once, optional)*

Cross-checks the depth backprojection math against Isaac's own ground-truth point cloud. If the mean nearest-neighbour distance is below 2 cm, conventions are correct. If you skipped this and results look wrong, start here.

```bash
python 00_verify_conventions.py                          # defaults to frame1
python 00_verify_conventions.py ../data/captures/frame3
```

Expected output:
```
ours -> gt nearest-neighbor distance: mean=0.00031  median=0.00021  max=0.01204
Looks good -- our conventions match Isaac Sim's ground truth.
```

---

### Step 1 — Build Per-View Point Clouds

Backprojects each frame's depth map into a point cloud in the camera's own local coordinate frame. Writes one `.pcd` file per frame to `outputs/per_view_pcd/`.

```bash
python 01_pcd_from_depth.py
```

---

### Step 2a — Merge Using Ground-Truth Poses

Transforms each per-view cloud into world space using the known extrinsics and concatenates them. Requires Step 1.

```bash
python 02a_registration_known_extrinsics.py
# Output: ../outputs/merged_scene_extrinsics.pcd
```

---

### Step 2b — Merge Using ICP *(no pose priors)*

Registers all views from geometry alone: FPFH features → RANSAC global alignment → point-to-plane ICP → pose graph optimisation. Requires Step 1. Slower than 2a — runtime scales quadratically with the number of views.

```bash
python 02b_registration_icp.py
# Output: ../outputs/merged_scene_icp.pcd
```

---

### Step 3 — Segment and Isolate the Object

Removes the ground plane (RANSAC), denoises the remainder (statistical + radius outlier removal), and extracts the largest cluster (DBSCAN). Works on the output of either 2a or 2b.

```bash
# Using ICP result (default)
python 03_plane_and_object_segmentation.py

# Using known-extrinsics result
python 03_plane_and_object_segmentation.py \
    --input ../outputs/merged_scene_extrinsics.pcd

# Output: ../outputs/object.pcd
```

Key tunable parameters:

| Parameter | Default | Description |
|---|---|---|
| `--plane-dist-thresh` | `0.008` m | RANSAC inlier threshold for plane fitting. Increase if the ground plane is not fully removed. |
| `--voxel` | `0.003` m | Final voxel downsample resolution for the isolated object cloud. |

---

### Step 4 — Estimate 6-DoF Pose

Estimates the object's pose from the isolated point cloud using PCA. Writes `pose.json` containing the 3×3 rotation matrix, 3D translation vector, and full 4×4 homogeneous transform.

```bash
python 04_pose_estimation_pca.py
# Output: ../outputs/pose.json
```

---

### Step 5 — Visualize

Opens an Open3D interactive window showing the isolated object cloud, the estimated pose as a coordinate frame, and the world-origin reference frame.

```bash
python 05_visualize.py
```

---

## Utility Scripts

### `view_pointcloud.py` — Inspect any point cloud at any stage

```bash
python view_pointcloud.py ../outputs/per_view_pcd/frame1.pcd
python view_pointcloud.py ../outputs/merged_scene_icp.pcd --voxel 0.005
python view_pointcloud.py ../outputs/object.pcd --uniform-color 0.8 0.2 0.2 --axis 0.1
```

Options:

| Flag | Description |
|---|---|
| `--axis <size>` | Draw a world-origin coordinate frame of the given size. Set to `0` to hide. |
| `--voxel <size>` | Downsample before viewing (useful for large merged clouds). |
| `--uniform-color R G B` | Paint the cloud a flat colour (0–1 range) instead of its own per-point colours. |

### `fly_cam.py` — Add a free-fly camera in Isaac Sim

Paste into Isaac Sim's Script Editor and run once. Creates a standard USD camera at `/World/FlyCam` navigable with RMB + WASD. Useful for manually framing the scene before setting up Replicator capture.

> This script runs inside Isaac Sim only — it is not a standalone Python script.

---

## Re-capturing Data with Isaac Sim *(optional)*

If you have Isaac Sim installed and want to capture your own data:

1. Open `scenes/objects_on_ground.usd` in Isaac Sim.
2. Use `fly_cam.py` (Script Editor) to navigate and frame the scene.
3. Configure the Replicator BasicWriter to output: `rgb`, `distance_to_image_plane`, `camera_params`, `pointcloud`.
4. Run the Replicator. Captured frames will be written to `data/captures/`.
5. Proceed from Step 0.

---

## Key Design Decisions

**Intrinsics from the projection matrix, not the aperture:** Avoids a systematic intrinsics error introduced by Isaac's aperture-fit conform policy. Confirmed empirically — see `utils_isaac_io.py` for a full explanation.

**Two registration methods side by side:** Having both a ground-truth method (2a) and an algorithmic method (2b) makes it easy to directly compare registration quality and understand where ICP-based alignment introduces error relative to the known-pose baseline.

**Step 0 as an explicit sanity check:** Camera coordinate conventions — sign of Y, sign of Z, row-major vs column-major matrices — are the most common source of silent, hard-to-debug errors in 3D perception pipelines. Formalising the check as a runnable script rather than a comment makes it repeatable and unambiguous.

---

## Dependencies

| Package | Purpose |
|---|---|
| `open3d` | Point cloud I/O, downsampling, normal estimation, FPFH features, RANSAC registration, ICP, pose graph optimisation, visualisation |
| `numpy` | Depth backprojection, matrix math, PCA |

No deep learning frameworks, no ROS, no CUDA required.

---

## Part of `general-perception`

This is one project within the [`general-perception`](https://github.com/Vihaan30g/general-perception) repository, which contains independent perception projects including:

- **[`pose_estimation/`](https://github.com/Vihaan30g/general-perception/tree/main/pose_estimation)** — this project
- **`ros2_ws/src/dino/`** — DINOv2-based feature extraction (ROS2 node)
- **`ros2_ws/src/sam/`** — Segment Anything Model integration (ROS2 node)
- **`isaac_sim_ws/`** — Shared Isaac Sim scenes and assets

Each sub-project is self-contained and can be used independently.
