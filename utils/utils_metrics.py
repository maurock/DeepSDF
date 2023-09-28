"""Metrics for evaluation of the model."""
import trimesh
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

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

# Adapted from Tatarchenko, et al. "What Do Single-view 3D Reconstruction Networks Learn?" 
def calculate_fscore(points_gt, points_pred, th: float=0.001):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    dist = distance.cdist(points_pred, points_gt)

    d1_min = np.min(dist, axis=1)
    d2_min = np.min(dist, axis=0)
    
    recall = np.sum(d2_min < th) / len(d2_min)
    precision = np.sum(d1_min < th) / len(d1_min)

    if recall + precision > 0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0

    return fscore, precision, recall

