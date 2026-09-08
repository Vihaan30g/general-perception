# scripts/04_plane_and_object_segmentation.py
"""
Step 3 - remove the support plane (table/ground) and isolate the object of
interest from a merged scene point cloud.

Works on the output of EITHER registration method:
    python 03_plane_and_object_segmentation.py --input outputs/merged_scene_extrinsics.pcd
    python 03_plane_and_object_segmentation.py --input outputs/merged_scene_icp.pcd
"""
import argparse
import os

import numpy as np
import open3d as o3d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../outputs/merged_scene_icp.pcd",
                     help="Merged scene pcd (from 03a or 03b)")
    ap.add_argument("--output", default="../outputs/object.pcd")
    ap.add_argument("--plane-dist-thresh", type=float, default=0.008,
                     help="RANSAC plane inlier distance threshold (meters)")
    ap.add_argument("--voxel", type=float, default=0.003,
                     help="Final voxel downsample size for the isolated object")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.input)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Loaded 0 points from {args.input}")

    # --- 1. Statistical outlier removal (clean registration artifacts) ---
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # --- 2. Plane (table) segmentation via RANSAC ---
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=args.plane_dist_thresh, ransac_n=3, num_iterations=1000)
    object_cloud = pcd.select_by_index(inliers, invert=True)  # everything NOT the plane

    # --- 3. Radius outlier removal on the remainder ---
    object_cloud, _ = object_cloud.remove_radius_outlier(nb_points=16, radius=0.02)

    # --- 4. DBSCAN clustering to isolate just the object (drop stray noise clusters) ---
    labels = np.array(object_cloud.cluster_dbscan(eps=0.015, min_points=20, print_progress=True))
    if labels.max() < 0:
        raise RuntimeError("No clusters found -- loosen eps/min_points")

    largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
    object_final = object_cloud.select_by_index(np.where(labels == largest_cluster)[0])

    # --- 5. Final voxel downsample (optional, for speed downstream) ---
    object_final = object_final.voxel_down_sample(voxel_size=args.voxel)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    o3d.io.write_point_cloud(args.output, object_final)
    print(f"Object isolated: {len(object_final.points)} points -> {args.output}")
