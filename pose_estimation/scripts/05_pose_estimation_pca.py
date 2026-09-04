# scripts/05_pose_estimation_pca.py
import numpy as np
import open3d as o3d
import json

pcd = o3d.io.read_point_cloud("output/object.pcd")
points = np.asarray(pcd.points)

# --- Translation: centroid ---
centroid = points.mean(axis=0)
centered = points - centroid

# --- Rotation: eigenvectors of the covariance matrix ---
cov = np.cov(centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)          # ascending eigenvalue order

order = np.argsort(eigvals)[::-1]               # descending: largest-spread axis first
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# Ensure a right-handed rotation matrix (determinant +1)
if np.linalg.det(eigvecs) < 0:
    eigvecs[:, -1] *= -1

R = eigvecs                                      # columns = object's principal axes in world frame

T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = centroid

print("Rotation:\n", R)
print("Translation:", centroid)

with open("output/pose.json", "w") as f:
    json.dump({"rotation": R.tolist(), "translation": centroid.tolist(),
               "transform_4x4": T.tolist()}, f, indent=2)
