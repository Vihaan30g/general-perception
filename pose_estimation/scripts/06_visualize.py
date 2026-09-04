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
