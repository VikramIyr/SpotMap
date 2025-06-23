import copy
import numpy as np
import open3d as o3d

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.2, 0.369, 0.588])
    target_temp.paint_uniform_color([0.820, 0.929, 0.976])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

def evaluate_registration(
    source          : o3d.geometry.PointCloud,
    target          : o3d.geometry.PointCloud, 
    transformation  : np.ndarray, 
    threshold       : float = 0.05,
    print_results   : bool = True,
):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_temp, target_temp, threshold
    )
    if print_results:
        print()
        print(f"Evaluation Results:")
        print(f"    Fitness: {evaluation.fitness:.6f}")
        print(f"    Inlier RMSE: {evaluation.inlier_rmse:.6f}")

    return evaluation.fitness, evaluation.inlier_rmse

def apply_point_to_point_icp(
    source          : o3d.geometry.PointCloud,
    target          : o3d.geometry.PointCloud, 
    threshold       : float = 0.05,
    trans_init      : np.ndarray = np.eye(4),
    max_iterations  : int = 50,
    print_results   : bool = True,
    visualize       : bool = True
):
    print()
    print(f"Applying Point-to-Point ICP with threshold {threshold} and max iterations {max_iterations}...")
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    if print_results:
        print()
        print(f"ICP Results:")
        print(f"    Fitness: {reg_p2p.fitness:.6f}")
        print(f"    Inlier RMSE: {reg_p2p.inlier_rmse:.6f}")
        print(f"    Number of Correspondences: {len(reg_p2p.correspondence_set)}")

    if visualize:
        draw_registration_result(source, target, reg_p2p.transformation)

    return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

def apply_point_to_plane_icp(
    source          : o3d.geometry.PointCloud,
    target          : o3d.geometry.PointCloud, 
    threshold       : float = 0.05,
    trans_init      : np.ndarray = np.eye(4),
    print_results   : bool = True,
    visualize       : bool = True
):
    print()
    print(f"Applying Point-to-Plane ICP with threshold {threshold}...")
    
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    if print_results:
        print()
        print(f"ICP Results:")
        print(f"    Fitness: {reg_p2l.fitness:.6f}")
        print(f"    Inlier RMSE: {reg_p2l.inlier_rmse:.6f}")
        print(f"    Number of Correspondences: {len(reg_p2l.correspondence_set)}")

    if visualize:
        draw_registration_result(source, target, reg_p2l.transformation)    
    
    return reg_p2l.transformation, reg_p2l.fitness, reg_p2l.inlier_rmse

def apply_robust_point_to_plane_icp(
    source          : o3d.geometry.PointCloud,
    target          : o3d.geometry.PointCloud, 
    threshold       : float = 0.03,
    trans_init      : np.ndarray = np.eye(4),
    sigma           : float = 0.05,
    print_results   : bool = True,
    visualize       : bool = True
):
    print()
    print(f"Applying Robust Point-to-Plane ICP with threshold {threshold} and Tukey kernel with sigma {sigma}...")

    loss = o3d.pipelines.registration.TukeyLoss(k=sigma) 

    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    )

    if print_results:
        print()
        print(f"ICP Results:")
        print(f"    Fitness: {reg_p2l.fitness:.6f}")
        print(f"    Inlier RMSE: {reg_p2l.inlier_rmse:.6f}")
        print(f"    Number of Correspondences: {len(reg_p2l.correspondence_set)}")

    if visualize:
        draw_registration_result(source, target, reg_p2l.transformation)
    
    return reg_p2l.transformation
    
def apply_colored_icp(
    source          : o3d.geometry.PointCloud,
    target          : o3d.geometry.PointCloud, 
    trans_init      : np.ndarray = np.eye(4),
    voxel_radius    : list = [0.04, 0.02, 0.01],
    max_iterations  : list = [50, 30, 14],
    print_results   : bool = True,
    visualize       : bool = True
):
    print()
    print(f"Applying Colored ICP with voxel radii {voxel_radius} and max iterations {max_iterations}...")

    current_transformation = trans_init

    for scale in range(3):
        iter = max_iterations[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print(f"    [1] Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print(f"    [2] Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print(f"    [3] Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation
        if print_results:
            print()
            print(f"    ICP Results:")
            print(f"        Fitness: {result_icp.fitness:.6f}")
            print(f"        Inlier RMSE: {result_icp.inlier_rmse:.6f}")
            print(f"        Number of Correspondences: {len(result_icp.correspondence_set)}")

    if visualize:
        draw_registration_result_original_color(source, target, current_transformation)

    return current_transformation, result_icp.fitness, result_icp.inlier_rmse