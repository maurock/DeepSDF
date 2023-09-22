"""Metrics for evaluation of the model."""
import trimesh
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

def compute_area(vertices, faces):
    """Compute the area of a mesh."""
    mesh = trimesh.Trimesh(vertices, faces)
    return mesh.area

# Borrowed from Xu, Wenqiang, et al. "Visual-Tactile Sensing for In-Hand Object Reconstruction." 
def earth_mover_distance(points_gt, point_pred):
    d = distance.cdist(points_gt, point_pred)

    assignment = linear_sum_assignment(d)
    emd = d[assignment].sum() / len(d)

    return emd

