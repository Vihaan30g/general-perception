# scripts/03_registration.py
import numpy as np
import open3d as o3d
import glob

VOXEL = 0.005   # 5mm — tune based on object scale

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

def refine_icp(src, tgt, voxel_size, init_transform):
    dist_thresh = voxel_size * 0.4
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, dist_thresh, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def pairwise_registration(src, tgt, voxel_size):
    src_down, src_fpfh = preprocess(src, voxel_size)
    tgt_down, tgt_fpfh = preprocess(tgt, voxel_size)
    coarse = global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size)
    fine = refine_icp(src_down, tgt_down, voxel_size, coarse.transformation)
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src_down, tgt_down, voxel_size * 1.5, fine.transformation)
    return fine.transformation, information

def full_registration(pcds, voxel_size):
    n = len(pcds)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for src_id in range(n):
        for tgt_id in range(src_id + 1, n):
            transform, info = pairwise_registration(pcds[src_id], pcds[tgt_id], voxel_size)
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
    files = sorted(glob.glob("output/per_view_pcd/*.pcd"))
    pcds = [o3d.io.read_point_cloud(f) for f in files]

    pose_graph = full_registration(pcds, VOXEL)

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
    o3d.io.write_point_cloud("output/merged_scene.pcd", merged)
    print(f"Merged cloud: {len(merged.points)} points -> output/merged_scene.pcd")
