# scripts/00_verify_conventions.py
"""
Step 0 (run ONCE, before trusting the rest of the pipeline).

Sanity-checks the camera math in utils_isaac_io.py against Isaac Sim's OWN
ground-truth world-space point cloud (pointcloud_<idx>.npy), written out by
the replicator at capture time.

If our depth-backprojection + extrinsics are correct, the cloud we build for
a frame and transform into world space should sit almost exactly on top of
Isaac's own pointcloud_<idx>.npy for that same frame.

Usage:
    python 00_verify_conventions.py                       # defaults to frame1
    python 00_verify_conventions.py data/captures/frame3
"""
import sys

import numpy as np
import open3d as o3d

from utils_isaac_io import (
    load_frame_raw,
    load_gt_pointcloud,
    compute_intrinsics,
    load_extrinsics,
    backproject_depth_to_camera_frame,
    transform_points,
)


def main(frame_dir):
    depth, rgb, cam_params = load_frame_raw(frame_dir)
    K, w, h = compute_intrinsics(cam_params)
    cam_to_world = load_extrinsics(cam_params)
    print(f"Frame: {frame_dir}  ({w}x{h})")
    print("Recovered camera position (world):", cam_to_world[:3, 3])

    pts_cam, _ = backproject_depth_to_camera_frame(depth, K)
    pts_world = transform_points(pts_cam, cam_to_world)

    ours = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world))
    gt_pts = load_gt_pointcloud(frame_dir)
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))

    print(f"ours: {len(ours.points):>7d} pts | bbox {ours.get_min_bound()} -> {ours.get_max_bound()}")
    print(f"gt:   {len(gt.points):>7d} pts | bbox {gt.get_min_bound()} -> {gt.get_max_bound()}")

    dists = np.asarray(ours.compute_point_cloud_distance(gt))
    print(f"\nours -> gt nearest-neighbor distance: "
          f"mean={dists.mean():.5f}  median={np.median(dists):.5f}  max={dists.max():.5f}")

    if dists.mean() > 0.02:
        print(
            "\n*** WARNING: mean distance is large relative to typical scene scale. "
            "This usually means a sign/convention mismatch. Things to try:\n"
            "  - flip the sign of `y` or `zc` in backproject_depth_to_camera_frame()\n"
            "  - drop the `.T` in load_extrinsics() (row/col-major mismatch)\n"
            "  - check cameraApertureOffset sign in compute_intrinsics()\n"
            "Re-run this script after each change until the distance drops near 0."
        )
    else:
        print("\nLooks good -- our conventions match Isaac Sim's ground truth.")

    ours.paint_uniform_color([1, 0, 0])  # red  = ours
    gt.paint_uniform_color([0, 1, 0])    # green = ground truth
    o3d.visualization.draw_geometries([ours, gt])


if __name__ == "__main__":
    frame_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/captures/frame1"
    main(frame_dir)
