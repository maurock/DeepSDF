import numpy as np
import torch
import results
import os
import meshplot as mp
import model.model_sdf as sdf_model
import argparse
import meshplot as mp
from utils import utils_deepsdf
import trimesh
mp.offline()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latent_codes_training(results_dict):
    """Return a list containing the latent code tensors optimised at training time"""
    latent_codes_list = []
    for i in range(results_dict['train']['best_latent_codes'].shape[0]):
        latent_codes_list.append(torch.from_numpy(results_dict['train']['best_latent_codes'][i]).to(device))
    return latent_codes_list

def main(args):
    folder = args.folder
    model = sdf_model.SDFModelMulti(num_layers=args.num_layers, no_skip_connections=args.no_skip_connections).to(device)

    # Load weights
    weights_path = os.path.join(os.path.dirname(results.__file__), 'runs_sdf', folder, 'weights.pt')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Load results dictionary
    results_dict_path = os.path.join(os.path.dirname(results.__file__), 'runs_sdf', folder, 'results.npy')
    results_dict = np.load(results_dict_path, allow_pickle=True).item()

    latent_codes_list = get_latent_codes_training(results_dict)
  
    coords, grad_size_axis = utils_deepsdf.get_volume_coords(args.resolution)

    for idx, latent in enumerate(latent_codes_list):
        sdf = utils_deepsdf.predict_sdf(latent, coords, model)
        vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)

        # save mesh using meshplot
        mesh_dir = os.path.join(os.path.dirname(results.__file__), 'runs_sdf', folder, 'meshes_training')
        if not os.path.exists(mesh_dir):
            os.mkdir(mesh_dir)
        mesh_path = os.path.join(mesh_dir, f'latent_{idx}.html')
        utils_deepsdf.save_meshplot(vertices, faces, mesh_path)

        # save mesh as obj
        obj_path = os.path.join(mesh_dir, f"mesh_{idx}.obj")
        trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type='obj')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, default='', help="Folder that contains the network parameters"
    )
    parser.add_argument(
        "--resolution", type=int, default=50, help="Folder that contains the network parameters"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number network layers"
    )
    parser.add_argument(
        "--no_skip_connections", default=False, action='store_true', help="Do not skip connections"
    )   
    args = parser.parse_args()

    main(args)