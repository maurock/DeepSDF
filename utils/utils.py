import numpy as np
import trimesh
from glob import glob
import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mesh_from_urdf(filepath):
    """
    Receives path to object index containing the .URDF and verts and faces (both np.array).
    Directory tree:
    - obj_idx
    |   - textured_objs
    |   |   - ...obj
    |- ...
    """
    total_objs = glob(os.path.join(filepath, 'textured_objs/*.obj'))
    verts = np.array([]).reshape((0,3))
    faces = np.array([]).reshape((0,3))

    mesh_list = []
    for obj_file in total_objs:
        mesh = _as_mesh(trimesh.load(obj_file))
        mesh_list.append(mesh)           
                
    verts_list = [mesh.vertices for mesh in mesh_list]
    faces_list = [mesh.faces for mesh in mesh_list]
    faces_offset = np.cumsum([v.shape[0] for v in verts_list], dtype=np.float32)   # num of faces per mesh
    faces_offset = np.insert(faces_offset, 0, 0)[:-1]            # compute offset for faces, otherwise they all start from 0
    verts = np.vstack(verts_list).astype(np.float32)
    faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)]).astype(np.float32)

    return verts, faces

def _as_mesh(scene_or_mesh):
    # Utils function to get a mesh from a trimesh.Trimesh() or trimesh.scene.Scene()
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def mesh_to_pointcloud(verts, faces, n_samples):
    """
    This method samples n points on a mesh. The number of samples for each face is weighted by its size. 

    Params:
        verts = vertices, np.array(n, 3)
        faces = faces, np.array(m, 3)
        n_samples: number of total samples
    
    Returns:
        pointcloud
    """
    mesh = trimesh.Trimesh(verts, faces)
    pointcloud, _ = trimesh.sample.sample_surface(mesh, n_samples)
    pointcloud = pointcloud.astype(np.float32)
    return pointcloud

def clamp(x, delta=torch.tensor([[0.1]]).to(device)):
    """Clamp function introduced in the paper DeepSDF.
    This returns a value in range [-delta, delta]. If x is within this range, it returns x, else one of the extremes.

    Args:
        x: prediction, torch tensor (batch_size, 1)
        delta: small value to control the distance from the surface over which we want to mantain metric SDF
    """
    maximum = torch.amax(torch.vstack((x, -delta)))
    minimum = torch.amin(torch.vstack((delta[0], maximum)))
    return minimum

def SDFLoss(sdf, predictions):
    """Loss function ontroducd in the paper DeepSDF"""
    return torch.abs(clamp(predictions) - clamp(sdf))

