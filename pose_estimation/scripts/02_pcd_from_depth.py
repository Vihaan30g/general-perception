# scripts/02_pcd_from_depth.py
"""
Step 2 - build one point cloud per captured frame, in that camera's OWN local
frame (camera-local coordinates), directly from the depth map + intrinsics.

This is the shared prerequisite for BOTH downstream registration methods:
    03a_registration_known_extrinsics.py  (uses ground-truth sim poses)
    03b_registration_icp.py               (estimates poses via FPFH+RANSAC+ICP)

Run this once. Per-view .pcd files are written to outputs/per_view_pcd/.
"""
import os

import numpy as np
import open3d as o3d

from utils_isaac_io import (
    find_frame_dirs,
    load_frame_raw,
    compute_intrinsics,
    backproject_depth_to_camera_frame,
)

DEPTH_TRUNC = 5.0   # meters - discard anything farther (also filters inf/sentinel values)
DEPTH_MIN = 0.01


def build_pcd_for_frame(frame_dir):
    depth, rgb, cam_params = load_frame_raw(frame_dir)
    K, width, height = compute_intrinsics(cam_params)

    pts_cam, valid = backproject_depth_to_camera_frame(
        depth, K, depth_min=DEPTH_MIN, depth_trunc=DEPTH_TRUNC)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_cam)

    if rgb is not None and rgb.shape[:2] == valid.shape:
        colors = rgb[valid].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


if __name__ == "__main__":
    frame_dirs = find_frame_dirs("data/captures")
    os.makedirs("outputs/per_view_pcd", exist_ok=True)

    for frame_dir in frame_dirs:
        name = os.path.basename(frame_dir)
        pcd = build_pcd_for_frame(frame_dir)
        out_path = f"outputs/per_view_pcd/{name}.pcd"
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"{name}: {len(pcd.points)} points (camera-local frame) -> {out_path}")
