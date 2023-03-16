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

"""Extract mesh from an already optimised latent code and network. 
Store the mesh in the same folder where the latent code is located."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    latent_code_path = args.latent_code_path
    latent_code = torch.load(latent_code_path, map_location=device)

    model = sdf_model.SDFModelMulti(num_layers=8, no_skip_connections=False, input_dim=args.latent_size + 3).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
   
    # Extract mesh obtained with the latent code optimised at inference
    coords, grad_size_axis = utils_deepsdf.get_volume_coords(args.resolution)
    coords = coords.to(device)
    coords_batches = torch.split(coords, 100000)

    sdf = utils_deepsdf.predict_sdf(latent_code, coords_batches, model)
    vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)

    # save mesh using meshplot
    output_path = os.path.dirname(latent_code_path)
    mesh_path = os.path.join(output_path, f"mesh_{datetime.now().strftime('%d_%m_%H%M%S')}.html")
    utils_deepsdf.save_meshplot(vertices, faces, mesh_path)

    # save mesh as obj
    obj_path = os.path.join(output_path, f"mesh_{datetime.now().strftime('%d_%m_%H%M%S')}.obj")
    trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type='obj')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent_code_path", type=str, default='', help="Path to the optimised latent code"
    )
    parser.add_argument(
        "--weights_path", type=str, default='', help="Path to the network parameters"
    )
    parser.add_argument(
        "--no_skip_connections", default=False, action='store_true', help="Do not skip connections"
    ) 
    parser.add_argument(
        "--resolution", type=int, default=50, help="Resolution of the extracted mesh"
    )
    args = parser.parse_args()

    main(args)

