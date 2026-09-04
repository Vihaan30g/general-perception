# scripts/03a_registration_known_extrinsics.py
"""
Step 3a - Method 1 (ground truth): build the merged scene using the exact
camera poses Isaac Sim used at capture time (camera_params_*.json ->
cameraViewTransform).

Since the pose of every frame is known exactly, no feature matching or ICP is
needed: we just transform each per-view point cloud (built by
02_pcd_from_depth.py, still in camera-local coordinates) into the world frame
with the known extrinsics and concatenate.

Requires: outputs/per_view_pcd/*.pcd from 02_pcd_from_depth.py
Output:   outputs/merged_scene_extrinsics.pcd
"""
import os

import open3d as o3d

from utils_isaac_io import find_frame_dirs, load_camera_params, load_extrinsics

VOXEL = 0.005   # 5mm final downsample - tune based on object/scene scale

if __name__ == "__main__":
    frame_dirs = find_frame_dirs("data/captures")
    merged = o3d.geometry.PointCloud()

    for frame_dir in frame_dirs:
        name = os.path.basename(frame_dir)
        pcd_path = f"outputs/per_view_pcd/{name}.pcd"
        pcd = o3d.io.read_point_cloud(pcd_path)  # camera-local frame

        cam_params = load_camera_params(frame_dir)
        cam_to_world = load_extrinsics(cam_params)

        pcd.transform(cam_to_world)  # now in world frame
        merged += pcd
        print(f"{name}: {len(pcd.points)} pts | camera position {cam_to_world[:3, 3]}")

    merged = merged.voxel_down_sample(VOXEL)
    os.makedirs("outputs", exist_ok=True)
    o3d.io.write_point_cloud("outputs/merged_scene_extrinsics.pcd", merged)
    print(f"\nMerged (known-extrinsics) cloud: {len(merged.points)} points "
          f"-> outputs/merged_scene_extrinsics.pcd")
