"""Metrics for evaluation of the model."""
import trimesh
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import point_cloud_utils as pcu

def calculate_error_area(vertices, faces, gt_area):
    """Compute the area of a mesh."""
    mesh = trimesh.Trimesh(vertices, faces)
    error = abs(mesh.area - gt_area) * 100 / gt_area

    return error

# Borrowed from Xu, Wenqiang, et al. "Visual-Tactile Sensing for In-Hand Object Reconstruction." 
def earth_mover_distance(points_gt, point_pred):
    d = distance.cdist(points_gt, point_pred)

    assignment = linear_sum_assignment(d)
    emd = d[assignment].sum() / len(d)

    return emd

def hausdorff_distance(points_gt, points_pred):
    distances_ab = distance.cdist(points_pred, points_gt)

    # Calculate the Hausdorff Distance by finding the maximum minimum distance in both directions
    hausdorff_ab = np.max(np.min(distances_ab, axis=0))
    hausdorff_ba = np.max(np.min(distances_ab, axis=1))
    hausdorff_distance = max(hausdorff_ab, hausdorff_ba)

    return hausdorff_distance


# # Adapted from Tatarchenko, et al. "What Do Single-view 3D Reconstruction Networks Learn?" 
# def calculate_fscore_point2point(points_gt, points_pred, th: float=0.001):
#     '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
#     dist = distance.cdist(points_pred, points_gt)

#     d1_min = np.min(dist, axis=1)
#     d2_min = np.min(dist, axis=0)
    
#     recall = np.sum(d2_min < th) / len(d2_min)
#     precision = np.sum(d1_min < th) / len(d1_min)

#     if recall + precision > 0:
#         fscore = 2 * recall * precision / (recall + precision)
#     else:
#         fscore = 0

#     return fscore, precision, recall


# Adapted from Tatarchenko, et al. "What Do Single-view 3D Reconstruction Networks Learn?" 
def calculate_fscore_point2surface(mesh_gt, mesh_pred, th: float=0.01):
    '''
    Calculates the F-score between point clouds and meshes.
    Precision measures the accuracy of the reconstruction by calculating the percentage 
    of reconstructed points within a certain distance to the ground truth. Recall measures
    the completeness of the reconstruction by counting the percentage of points on the 
    ground truth that lie within a certain distance to the reconstruction. The strictness 
    of the F-score can be controlled by varying the distance threshold d relative to the
    object size.
    '''

    points_gt = trimesh.sample.sample_surface(mesh_gt, 20000)[0]
    points_pred = trimesh.sample.sample_surface(mesh_pred, 20000)[0]

    # Threshold value equal to maximum size * th
    th_value = max([points_gt[:,i].max() - points_gt[:,i].min() for i in range(0,3)]) * th

    sdf_pred_point_to_gt_mesh, _, _  = pcu.signed_distance_to_mesh(points_pred, mesh_gt.vertices, mesh_gt.faces)
    sdf_gt_point_to_pred_mesh, _, _  = pcu.signed_distance_to_mesh(points_gt, mesh_pred.vertices, mesh_pred.faces)

    recall = len(sdf_gt_point_to_pred_mesh[sdf_gt_point_to_pred_mesh < th_value]) / len(sdf_gt_point_to_pred_mesh)
    precision = len(sdf_pred_point_to_gt_mesh[sdf_pred_point_to_gt_mesh < th_value]) / len(sdf_pred_point_to_gt_mesh)

    if recall + precision > 0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0

    return fscore, precision, recall

