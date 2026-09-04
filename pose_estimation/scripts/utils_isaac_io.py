# scripts/utils_isaac_io.py
"""
Shared IO / camera-math helpers for the Isaac Sim -> Open3D pose-estimation pipeline.

Data layout (per frame folder, e.g. data/captures/frame1/):
    camera_params_<idx>.json          - Replicator BasicWriter camera params
    distance_to_image_plane_<idx>.npy - depth map (H, W) float32 -- this is a
                                         "z-depth" along the camera's optical axis,
                                         NOT a euclidean ray length
    rgb_<idx>.png                     - color image (H, W, 3) uint8
    pointcloud_<idx>.npy              - Replicator's OWN world-space point cloud.
                                         This is ground truth, used only by
                                         00_verify_conventions.py to sanity-check
                                         the math below.
    pointcloud_rgb_/_normals_/_instance_/_semantic_<idx>.npy - per-point attributes
                                         aligned with pointcloud_<idx>.npy (unused
                                         by this pipeline)
    metadata.txt                      - replicator run metadata (unused)

Camera convention (USD / Omniverse -- confirmed against Isaac Sim 6.0.1 output):
    - Right-handed, Y-up world.
    - Camera-local frame is OpenGL-style: X-right, Y-up, camera looks down -Z.
    - `cameraViewTransform` is USD's row-major, ROW-VECTOR "view" matrix, i.e.
          p_world_row (1x4) @ M  ==  p_camera_row (1x4)
      To use it as a normal column-vector transform we transpose it first:
          world_to_cam (col-vector form) = M.T
          cam_to_world                   = inv(M.T)
      (You can see this in the raw JSON: the translation-looking values sit in
      the LAST ROW of the 4x4, which is the signature of a row-vector matrix --
      a column-vector matrix would have translation in the last COLUMN.)
    - `distance_to_image_plane` is the depth along the optical axis, so in
      camera-local coordinates z_cam = -depth (camera looks down -Z).

IMPORTANT: run scripts/00_verify_conventions.py once before trusting the rest
of the pipeline. It cross-checks the point cloud built here against Isaac's
own `pointcloud_<idx>.npy` and will tell you loudly if a sign is flipped.
"""
import os
import re
import glob
import json

import numpy as np
import open3d as o3d


def find_frame_dirs(captures_root="data/captures"):
    """Return frame directories sorted naturally (frame1, frame2, ..., frame10)."""
    dirs = [
        os.path.join(captures_root, d)
        for d in os.listdir(captures_root)
        if os.path.isdir(os.path.join(captures_root, d))
    ]

    def frame_num(d):
        m = re.search(r"(\d+)$", os.path.basename(d))
        return int(m.group(1)) if m else 0

    return sorted(dirs, key=frame_num)


def _find_one(frame_dir, pattern):
    matches = sorted(glob.glob(os.path.join(frame_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {frame_dir}")
    return matches[0]


def load_camera_params(frame_dir):
    path = _find_one(frame_dir, "camera_params_*.json")
    with open(path) as f:
        return json.load(f)


def load_depth(frame_dir):
    path = _find_one(frame_dir, "distance_to_image_plane_*.npy")
    return np.load(path).astype(np.float64)


def load_rgb(frame_dir):
    path = _find_one(frame_dir, "rgb_*.png")
    img = o3d.io.read_image(path)
    return np.asarray(img)


def load_gt_pointcloud(frame_dir):
    """Replicator's own world-space point cloud -- verification purposes only."""
    path = _find_one(frame_dir, "pointcloud_[0-9]*.npy")
    return np.load(path)


def load_frame_raw(frame_dir):
    depth = load_depth(frame_dir)
    rgb = load_rgb(frame_dir)
    cam_params = load_camera_params(frame_dir)
    return depth, rgb, cam_params


def compute_intrinsics(cam_params):
    """
    Derive pixel-space intrinsics from Isaac's physical camera parameters
    (standard Omniverse/Isaac Sim replicator formula):
        fx = width  * focal_length / horizontal_aperture
        fy = height * focal_length / vertical_aperture
    """
    width, height = cam_params["renderProductResolution"]
    focal_length = cam_params["cameraFocalLength"]
    h_aperture, v_aperture = cam_params["cameraAperture"]
    h_offset, v_offset = cam_params.get("cameraApertureOffset", [0.0, 0.0])

    fx = width * focal_length / h_aperture
    fy = height * focal_length / v_aperture
    cx = width / 2.0 - h_offset
    cy = height / 2.0 - v_offset

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])
    return K, int(width), int(height)


def load_extrinsics(cam_params):
    """
    Returns the 4x4 camera-to-world transform (column-vector convention:
    p_world = T @ p_camera_homogeneous), derived from `cameraViewTransform`.
    See the module docstring for why the transpose+invert is needed.
    """
    M = np.asarray(cam_params["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    world_to_cam = M.T
    cam_to_world = np.linalg.inv(world_to_cam)
    return cam_to_world


def backproject_depth_to_camera_frame(depth, K, depth_min=0.01, depth_trunc=5.0):
    """
    Backproject a z-depth map into the camera's OWN local frame, using Isaac's
    OpenGL-style camera convention (X-right, Y-up, looking down -Z).

    Returns:
        pts_valid: (N, 3) float64 points in camera-local coordinates
        valid:     (H, W) bool mask of which pixels were used (for aligning RGB)
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))  # both shape (h, w)
    z = depth
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_trunc)

    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy   # image row grows downward, camera Y is up -> flip
    zc = -z                  # camera looks down -Z; depth is a positive magnitude

    pts = np.stack([x, y, zc], axis=-1)
    pts_valid = pts[valid]
    return pts_valid, valid


def transform_points(points, T):
    """Apply a 4x4 homogeneous transform T to an (N, 3) point array."""
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    return (T @ pts_h.T).T[:, :3]
