# scripts/06_visualize.py
"""
Step 6 - visualize the isolated object together with its estimated pose
(coordinate frame) and a world-origin reference frame.
"""
import argparse
import json

import numpy as np
import open3d as o3d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="../outputs/object.pcd")
    ap.add_argument("--pose", default="../outputs/pose.json")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.object)
    with open(args.pose) as f:
        pose = json.load(f)
    T = np.array(pose["transform_4x4"])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    frame.transform(T)

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)  # origin reference

    o3d.visualization.draw_geometries([pcd, frame, world_frame])
