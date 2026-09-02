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
