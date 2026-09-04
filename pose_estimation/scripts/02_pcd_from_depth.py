# scripts/02_pcd_from_depth.py
import numpy as np
import open3d as o3d
import json, os

def load_view(view_dir):
    depth = np.load(os.path.join(view_dir, "depth.npy"))
    rgb = o3d.io.read_image(os.path.join(view_dir, "rgb.png"))
    with open(os.path.join(view_dir, "intrinsics.json")) as f:
        meta = json.load(f)
    K = np.array(meta["K"])
    return depth, np.asarray(rgb), K, meta["width"], meta["height"]

def depth_to_pointcloud(depth, K, rgb=None, depth_trunc=3.0, depth_min=0.05):
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid = (z > depth_min) & (z < depth_trunc)

    x = (i - cx) * z / fx
    y = (j - cy) * z / fy      # OpenCV/pinhole convention: x-right, y-down, z-forward

    pts = np.stack([x, y, z], axis=-1)[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if rgb is not None:
        colors = rgb[valid].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

if __name__ == "__main__":
    view_dirs = sorted(
        os.path.join("data/captures", d) for d in os.listdir("data/captures")
    )
    os.makedirs("output/per_view_pcd", exist_ok=True)

    for view_dir in view_dirs:
        depth, rgb, K, w, h = load_view(view_dir)
        pcd = depth_to_pointcloud(depth, K, rgb)
        name = os.path.basename(view_dir)
        o3d.io.write_point_cloud(f"output/per_view_pcd/{name}.pcd", pcd)
        print(f"{name}: {len(pcd.points)} points")
