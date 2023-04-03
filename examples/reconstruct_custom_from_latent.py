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

"""Extract mesh from an already optimised latent code and network. 
Store the mesh in the same folder where the latent code is located."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    if args.latent_code_path != '':
        latent_code_path = args.latent_code_path
        latent_code = torch.load(latent_code_path, map_location=device)
        obj_idx = 99999  # placeholder index
    else:
        # Load paths
        str2int_path = os.path.join(os.path.dirname(results.__file__), 'idx_str2int_dict.npy')
        results_dict_path = os.path.join(os.path.dirname(runs_sdf.__file__), args.folder_sdf, 'results.npy')
        
        # Load dictionaries
        str2int_dict = np.load(str2int_path, allow_pickle=True).item()
        results_dict = np.load(results_dict_path, allow_pickle=True).item()

        # Get object index in the results dictionary
        obj_idx = str2int_dict[args.obj_id]  # index in collected latent vector
        # Get the latent code optimised during training
        latent_code = results_dict['train']['best_latent_codes'][obj_idx]
        latent_code = torch.tensor(latent_code).to(device)

    weights = os.path.join(os.path.dirname(runs_sdf.__file__), args.folder_sdf, 'weights.pt')

    latent_size = latent_code.shape[0]

    model = sdf_model.SDFModelMulti(
        num_layers=8, 
        no_skip_connections=False, 
        latent_size=latent_size, 
        inner_dim=args.inner_dim,
        positional_encoding_embeddings=args.positional_encoding_embeddings).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
   
    # Extract mesh obtained with the latent code optimised at inference
    coords, grad_size_axis = utils_deepsdf.get_volume_coords(args.resolution)
    coords = coords.to(device)

    coords_batches = torch.split(coords, 100000)

    sdf = utils_deepsdf.predict_sdf(latent_code, coords_batches, model)
    try:
        vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)
    except:
        print('Mesh extraction failed')
        return
    
    # save mesh using meshplot
    mesh_dir = os.path.join(os.path.dirname(runs_sdf.__file__), args.folder_sdf, 'meshes_training')
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)
    mesh_path = os.path.join(mesh_dir, f'latent_{obj_idx}.html')
    utils_deepsdf.save_meshplot(vertices, faces, mesh_path)

    # save mesh as obj
    obj_path = os.path.join(mesh_dir, f"mesh_{obj_idx}.obj")
    trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type='obj')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent_code_path", type=str, default='', help="Path to the optimised latent code"
    )
    parser.add_argument(
        "--folder_sdf", type=str, default='', help="Path to the network parameters"
    )
    parser.add_argument(
        "--obj_id", type=str, default='', help="Object ID to reconstruct form ShapeNetCore the format category/instance ID, e.g. 02942699/6d036fd1c70e5a5849493d905c02fa86"
    )
    parser.add_argument(
        "--no_skip_connections", default=False, action='store_true', help="Do not skip connections"
    ) 
    parser.add_argument(
        "--resolution", type=int, default=50, help="Resolution of the extracted mesh"
    )
    parser.add_argument(
        "--inner_dim", type=int, default=512, help="Inner dimensions of the network"
    )
    parser.add_argument(
        "--positional_encoding_embeddings", type=int, default=0, help="Number of embeddingsto use for positional encoding. If 0, no positional encoding is used."
    )
    args = parser.parse_args()

    main(args)