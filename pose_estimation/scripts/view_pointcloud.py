# scripts/view_pointcloud.py
"""
Generic point-cloud viewer -- use this at ANY stage of the pipeline to
eyeball a .pcd/.ply/.xyz file.

Usage:
    python view_pointcloud.py outputs/per_view_pcd/frame1.pcd
    python view_pointcloud.py outputs/merged_scene_icp.pcd --axis 0.2
    python view_pointcloud.py outputs/object.pcd --uniform-color 0.8 0.2 0.2
    python view_pointcloud.py outputs/merged_scene_extrinsics.pcd --voxel 0.005
""" 
import argparse

import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to a point cloud file (.pcd, .ply, .xyz, ...)")
    ap.add_argument("--axis", type=float, default=0.1,
                     help="Size of the world coordinate frame (0 to hide)")
    ap.add_argument("--voxel", type=float, default=0.0,
                     help="Optional voxel downsample size before viewing")
    ap.add_argument("--uniform-color", type=float, nargs=3, default=None,
                     metavar=("R", "G", "B"),
                     help="Paint the cloud a flat color (0-1 range) instead of its own colors")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Loaded 0 points from {args.path} -- check the path/format.")

    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)

    if args.uniform_color is not None:
        pcd.paint_uniform_color(args.uniform_color)

    geoms = [pcd]
    if args.axis > 0:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis))

    print(f"{args.path}: {len(pcd.points)} points | "
          f"bbox {pcd.get_min_bound()} -> {pcd.get_max_bound()}")
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
