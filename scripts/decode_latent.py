import torch
import os
import meshplot as mp
mp.offline()
import model.sdf_model as sdf_model
import argparse
from tqdm import tqdm 
import utils.utils as utils
from datetime import datetime

"""Extract mesh from an already optimised latent code and network. 
Store the mesh in the same folder where the latent code is located."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    latent_code_path = args.latent_code_path
    latent_code = torch.load(latent_code_path, map_location=device)

    model = sdf_model.SDFModelMulti(num_layers=8, no_skip_connections=False).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
   
    # Extract mesh obtained with the latent code optimised at inference
    coords, grad_size_axis = utils.get_volume_coords(args.resolution)

    sdf = utils.predict_sdf(latent_code, coords, model)
    vertices, faces = utils.extract_mesh(grad_size_axis, sdf)

    # save mesh using meshplot
    output_path = os.path.dirname(latent_code_path)
    mesh_path = os.path.join(output_path, f"mesh_{datetime.now().strftime('%d_%m_%H%M%S')}.html")
    utils.save_meshplot(vertices, faces, mesh_path)


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

