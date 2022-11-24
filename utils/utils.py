import numpy as np
import trimesh
from glob import glob
import os
import torch
from copy import deepcopy
import pybullet as pb
import meshplot as mp
import skimage

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
    """L1 function introduced in the paper DeepSDF """
    return torch.abs(clamp(predictions) - clamp(sdf))

def SDFLoss_multishape(sdf, prediction, x_latent, sigma):
    """Loss function introduced in the paper DeepSDF for multiple shapes."""
    l1 = torch.mean(torch.abs(prediction - sdf))
    l2 = sigma**2 * torch.mean(torch.linalg.norm(x_latent, dim=1, ord=2, dtype=torch.float32))
    loss = l1 + l2
    #print(f'Loss prediction: {l1:.3f}, Loss regulariser: {l2:.3f}')
    return loss

def rotate_vertices(vertices, rot=[np.pi / 2, 0, 0]):
    """Rotate vertices by 90 deg around the x-axis. """
    new_verts = deepcopy(vertices)
    # Rotate object
    rot_Q_obj = pb.getQuaternionFromEuler(rot)
    rot_M_obj = np.array(pb.getMatrixFromQuaternion(rot_Q_obj)).reshape(3, 3)
    new_verts = np.einsum('ij,kj->ik', rot_M_obj, new_verts).transpose(1, 0)
    return new_verts

def generate_latent_codes(latent_size, samples_dict):
    """Generate a random latent codes for each shape form a Gaussian distribution
    Returns:
        - latent_codes: np.array, shape (num_shapes, latent_size)
        - dict_latent_codes: key: obj_index, value: corresponding idx in the latent_codes array. 
                                  e.g.  latent_codes = ([ [1, 2, 3], [7, 8, 9] ])
                                        dict_latent_codes[345] = 0, the obj that has index 345 refers to 
                                        the 0-th latent code.
    """
    latent_codes = torch.tensor([]).reshape(0, latent_size).to(device)
    dict_latent_codes = dict()
    for i, obj_idx in enumerate(list(samples_dict.keys())):
        dict_latent_codes[obj_idx] = i
        latent_code = torch.normal(0, 0.01, size = (1, latent_size), dtype=torch.float32).to(device)
        latent_codes = torch.vstack((latent_codes, latent_code))
    latent_codes.requires_grad_(True)
    return latent_codes, dict_latent_codes

def _weight_histograms_linear(writer, step, weights, name_layer):
    # flatten weights for tensorboard
    flattened_weights = weights.flatten()
    tag = f"layer_{name_layer}"
    writer.add_histogram(tag, flattened_weights, global_step=step)

def weight_to_tensorboard(writer, step, model):
    """visualize weights in tensorboard as histograms"""
    # Iterate over all model layers
    for name, param in model.named_parameters():
        _weight_histograms_linear(writer, step, param.data, name)

def latent_to_tensorboard(writer, step, latent_codes):
    """visualize latent_codes in tensorboard as histograms"""
    # Iterate over all model layers
    for i, latent_code in enumerate(latent_codes):
        tag = f"latent_code_{i}"
        writer.add_histogram(tag, latent_code, global_step=step)
    for i, latent_code_grad in enumerate(latent_codes.grad):
        tag = f"grad_latent_code_{i}"
        writer.add_histogram(tag, latent_code_grad, global_step=step)

def model_graph_to_tensorboard(train_loader, model, writer, generate_xy):
    batch = next(iter(train_loader))
    x, y, latent_codes_indexes_batch, latent_codes_batch = generate_xy(batch)
    writer.add_graph(model, x)

def get_volume_coords(resolution = 50):
    """Get threedimensional vector (M, N, P) accoridng to the desired resolutions."""
    grid_values = torch.arange(-1, 1, float(1/resolution)).to(device) # 50 resolution -> 1/50 
    grad_size_axis = grid_values.shape[0]
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)
    return coords, grad_size_axis

def save_meshplot(vertices, faces, path):
    mp.plot(vertices, faces, c=vertices[:, 2], filename=path)

def predict_sdf(latent, coords, model):
    sdf = torch.tensor([], dtype=torch.float32).view(0, 1).to(device)
    coords = coords.clone().to(device)
    coords_batches = torch.split(coords, 1000000)
    for coords in coords_batches:
        latent_tile = torch.tile(latent, (coords.shape[0], 1))
        coords_latent = torch.hstack((latent_tile, coords)).to(device)
        model.eval()
        with torch.no_grad():
            sdf_batch = model(coords_latent)
        sdf = torch.vstack((sdf, sdf_batch))        
    return sdf

def extract_mesh(grad_size_axis, sdf):
    grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)
    return vertices, faces