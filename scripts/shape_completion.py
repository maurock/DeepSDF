import torch
import os
import model.model_sdf as sdf_model
from utils import utils_deepsdf
import trimesh
from results import runs_sdf
import numpy as np
import config_files
import yaml
import data.ShapeNetCoreV2 as ShapeNetCoreV2
from utils import utils_mesh
import pybullet as pb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
"""Infer and reconstruct mesh from a partial point cloud.
Store the mesh in the same folder where the latent code is located."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_params(cfg):
    """Read the settings from the settings.yaml file. These are the settings used during training."""
    training_settings_path = os.path.join(os.path.dirname(runs_sdf.__file__),  cfg['folder_sdf'], 'settings.yaml') 
    with open(training_settings_path, 'rb') as f:
        training_settings = yaml.load(f, Loader=yaml.FullLoader)

    return training_settings


def reconstruct_object(cfg, latent_code, obj_idx, model, coords_batches, grad_size_axis): 
    """
    Reconstruct the object from the latent code and save the mesh.
    Meshes are stored as .obj files under the same folder cerated during training, for example:
    - runs_sdf/<datetime>/meshes_training/mesh_0.obj
    """
    sdf = utils_deepsdf.predict_sdf(latent_code, coords_batches, model)
    try:
        vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)
    except:
        print('Mesh extraction failed')
        return
    
    # save mesh as obj
    mesh_dir = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'meshes_training')
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)
    obj_path = os.path.join(mesh_dir, f"mesh_{obj_idx}.obj")
    trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type='obj')


def generate_partial_pointcloud(cfg):
    """Load mesh and generate partial point cloud. The ratio of the visible bounding box is defined in the config file.
    Args:
        cfg: config file
    Return:
        samples: np.array, shape (N, 3), where N is the number of points in the partial point cloud.
        """
    # Load mesh
    obj_path = os.path.join(os.path.dirname(ShapeNetCoreV2.__file__), cfg['obj_ids'], 'models', 'model_normalized.obj')
    mesh_original = utils_mesh._as_mesh(trimesh.load(obj_path))

    # In Shapenet, the front is the -Z axis with +Y still being the up axis. 
    # Rotate objects to align with the canonical axis. 
    mesh = utils_mesh.shapenet_rotate(mesh_original)

    # Sample on the object surface
    samples = np.array(trimesh.sample.sample_surface(mesh, 10000)[0])

    # Infer object bounding box and collect samples on the surface of the objects when the x-axis is lower than a certain threshold t.
    # This is to simulate a partial point cloud.
    t = [cfg['x_axis_ratio_bbox'], cfg['y_axis_ratio_bbox'], cfg['z_axis_ratio_bbox']]

    v_min, v_max = mesh.bounds

    for i in range(3):
        t_max = v_min[i] + t[i] * (v_max[i] - v_min[i])
        samples = samples[samples[:, i] < t_max]
    
    return samples


def main(cfg):
    model_settings = read_params(cfg)

    # Set directory and paths
    model_dir = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'])

    inference_dir = os.path.join(model_dir, f"infer_latent_{datetime.now().strftime('%d_%m_%H%M%S')}")
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)

    output_mesh_path = os.path.join(inference_dir, 'output_mesh.obj')

    # Set tensorboard writer
    writer = SummaryWriter(log_dir=inference_dir, filename_suffix='inference_tensorboard')

    # Load the model
    weights = os.path.join(model_dir, 'weights.pt')

    model = sdf_model.SDFModel(
        num_layers=model_settings['num_layers'], 
        skip_connections=model_settings['latent_size'], 
        latent_size=model_settings['latent_size'], 
        inner_dim=model_settings['inner_dim']).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
   
    # Define coordinates for mesh extraction
    coords, grad_size_axis = utils_deepsdf.get_volume_coords(cfg['resolution'])
    coords = coords.to(device)

    # Split coords into batches because of memory limitations
    coords_batches = torch.split(coords, 100000)

    # Generate partial point cloud
    pointcloud = generate_partial_pointcloud(cfg)

    # Save partial pointcloud
    pointcloud_path = os.path.join(inference_dir, 'partial_pointcloud.npy')
    np.save(pointcloud_path, pointcloud)

    # Generate torch tensors of zeros that has the same dimension as pointcloud
    pointcloud = torch.tensor(pointcloud, dtype=torch.float32).to(device)
    sdf_gt = torch.zeros_like(pointcloud[:, 0]).view(-1, 1).to(device)

    # Get the average optimised latent code
    results_path = os.path.join(model_dir, 'results.npy')
    results = np.load(results_path, allow_pickle=True).item()
    latent_code = results['best_latent_codes']
    # Get average latent code (across dimensions)
    latent_code = torch.mean(torch.tensor(latent_code, dtype=torch.float32), dim=0).to(device)
    latent_code.requires_grad = True
    
    # Infer latent code
    best_latent_code = model.infer_latent_code(cfg, pointcloud, sdf_gt, writer, latent_code)

    # Extract mesh obtained with the latent code optimised at inference
    sdf = utils_deepsdf.predict_sdf(best_latent_code, coords_batches, model)
    vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)
    output_mesh = utils_mesh._as_mesh(trimesh.Trimesh(vertices, faces))

    # Save mesh
    trimesh.exchange.export.export_mesh(output_mesh, output_mesh_path, file_type='obj')


if __name__ == '__main__':

    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'shape_completion.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)