"""Metrics for evaluation of the model."""
import trimesh
import torch
import numpy as np
from scipy.spatial import Delaunay

def compute_area(vertices, faces):
    """Compute the area of a mesh."""
    mesh = trimesh.Trimesh(vertices, faces)
    return mesh.area

