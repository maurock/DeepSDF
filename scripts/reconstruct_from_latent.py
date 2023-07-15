import torch
import os
import meshplot as mp
mp.offline()
import model.model_sdf as sdf_model
import argparse
from tqdm import tqdm 
from utils import utils_deepsdf
from datetime import datetime
import trimesh
from results import runs_sdf
import results
import numpy as np
import configs
import yaml
"""Extract mesh from an already optimised latent code and network. 
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


def main(cfg):
    training_settings = read_params(cfg)

    # Load the model
    weights = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'weights.pt')

    model = sdf_model.SDFModel(
        num_layers=training_settings['num_layers'], 
        skip_connections=training_settings['latent_size'], 
        latent_size=training_settings['latent_size'], 
        inner_dim=training_settings['inner_dim']).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
   
    # Extract mesh obtained with the latent code optimised at inference
    coords, grad_size_axis = utils_deepsdf.get_volume_coords(cfg['resolution'])
    coords = coords.to(device)

    # Split coords into batches because of memory limitations
    coords_batches = torch.split(coords, 100000)
    
    # Load paths
    str2int_path = os.path.join(os.path.dirname(results.__file__), 'idx_str2int_dict.npy')
    results_dict_path = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'results.npy')
    
    # Load dictionaries
    str2int_dict = np.load(str2int_path, allow_pickle=True).item()
    results_dict = np.load(results_dict_path, allow_pickle=True).item()

    for obj_id_path in cfg['obj_ids']:
        # Get object index in the results dictionary
        obj_idx = str2int_dict[obj_id_path]  # index in collected latent vector
        # Get the latent code optimised during training
        latent_code = results_dict['best_latent_codes'][obj_idx]
        latent_code = torch.tensor(latent_code).to(device)

        reconstruct_object(cfg, latent_code, obj_idx, model, coords_batches, grad_size_axis)


if __name__ == '__main__':

    cfg_path = os.path.join(os.path.dirname(configs.__file__), 'reconstruct_from_latent.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)