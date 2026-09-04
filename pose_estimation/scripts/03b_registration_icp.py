# scripts/03b_registration_icp.py
"""
Step 3b - Method 2 (algorithmic): register the per-view point clouds purely
from geometry, WITHOUT using the simulation's ground-truth camera poses.

Pipeline per pair of views:
    1. Voxel-downsample + estimate normals + compute FPFH features
    2. Global registration: RANSAC based on FPFH feature matching (coarse alignment,
       no initial guess needed)
    3. Local refinement: point-to-plane ICP
    4. Build a pose graph (odometry edges between consecutive frames, loop-closure
       edges between all other pairs) and run global pose graph optimization

Requires: outputs/per_view_pcd/*.pcd from 02_pcd_from_depth.py
Output:   outputs/merged_scene_icp.pcd
"""
import os
import glob

import numpy as np
import open3d as o3d

VOXEL = 0.005   # 5mm - tune based on object/scene scale


def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh


def global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size):
    dist_thresh = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        dist_thresh,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )
    return result


def refine_icp(src_down, tgt_down, voxel_size, init_transform):
    """src_down/tgt_down already have normals from preprocess()."""
    dist_thresh = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down, dist_thresh, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def full_registration(pcds, voxel_size):
    n = len(pcds)

    # Preprocess each cloud once and reuse across all pairs.
    downs, fpfhs = [], []
    for pcd in pcds:
        d, f = preprocess(pcd, voxel_size)
        downs.append(d)
        fpfhs.append(f)

    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for src_id in range(n):
        for tgt_id in range(src_id + 1, n):
            coarse = global_registration(
                downs[src_id], downs[tgt_id], fpfhs[src_id], fpfhs[tgt_id], voxel_size)
            fine = refine_icp(downs[src_id], downs[tgt_id], voxel_size, coarse.transformation)
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                downs[src_id], downs[tgt_id], voxel_size * 1.5, fine.transformation)
            transform = fine.transformation

            print(f"  pair ({src_id},{tgt_id}): "
                  f"fitness={fine.fitness:.3f} rmse={fine.inlier_rmse:.5f}")

            if tgt_id == src_id + 1:      # sequential (odometry) edge
                odometry = transform @ odometry
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src_id, tgt_id, transform, info, uncertain=False))
            else:                          # loop-closure edge
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src_id, tgt_id, transform, info, uncertain=True))
    return pose_graph


if __name__ == "__main__":
    files = sorted(glob.glob("../outputs/per_view_pcd/*.pcd"))
    if not files:
        raise RuntimeError("No per-view pcds found. Run 02_pcd_from_depth.py first.")
    pcds = [o3d.io.read_point_cloud(f) for f in files]

    print("Running pairwise global + ICP registration...")
    pose_graph = full_registration(pcds, VOXEL)

    print("\nOptimizing pose graph...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=VOXEL * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    merged = o3d.geometry.PointCloud()
    for i, pcd in enumerate(pcds):
        pcd_t = pcd.transform(pose_graph.nodes[i].pose)
        merged += pcd_t

    merged = merged.voxel_down_sample(VOXEL)
    os.makedirs("outputs", exist_ok=True)
    o3d.io.write_point_cloud("../outputs/merged_scene_icp.pcd", merged)
    print(f"\nMerged (ICP) cloud: {len(merged.points)} points -> outputs/merged_scene_icp.pcd")
