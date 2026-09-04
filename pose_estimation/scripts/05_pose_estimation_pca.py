# scripts/05_pose_estimation_pca.py
"""
Step 5 - estimate the object's 6D pose (position + orientation) from its
isolated point cloud using PCA: centroid gives translation, eigenvectors of
the point covariance give the object's principal axes (rotation).
"""
import argparse
import json

import numpy as np
import open3d as o3d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/object.pcd")
    ap.add_argument("--output", default="outputs/pose.json")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise RuntimeError(f"Loaded 0 points from {args.input}")

    # --- Translation: centroid ---
    centroid = points.mean(axis=0)
    centered = points - centroid

    # --- Rotation: eigenvectors of the covariance matrix ---
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending eigenvalue order

    order = np.argsort(eigvals)[::-1]        # descending: largest-spread axis first
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Ensure a right-handed rotation matrix (determinant +1)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1

    R = eigvecs   # columns = object's principal axes in world frame

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centroid

    print("Rotation:\n", R)
    print("Translation:", centroid)

    with open(args.output, "w") as f:
        json.dump({"rotation": R.tolist(), "translation": centroid.tolist(),
                   "transform_4x4": T.tolist()}, f, indent=2)
    print(f"Pose written -> {args.output}")
