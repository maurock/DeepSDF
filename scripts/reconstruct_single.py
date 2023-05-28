import torch
import results.runs_sdf as runs
import os
import model.model_sdf as sdf_model
import argparse
import trimesh
import numpy as np
import results
import skimage.measure
"""
Test model
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_mesh(data, sdf_prediction):
    """Extract mesh using the marching cubes algorithm.
    Args:
        data: latent code + 3D points in a volume
        sdf: prediction
    Returns:
        mesh
    """
    sdf_inside = data[(sdf_prediction.view(-1)>-0.01) & (sdf_prediction.view(-1)<0.01)]
    #sdf_inside = data[(sdf_prediction.view(-1)<0.01)]
    mesh = trimesh.voxel.ops.points_to_marching_cubes(sdf_inside.detach().cpu().numpy(), pitch=0.02)
    return mesh

def extract_mesh_skimage(sdf_prediction):
    verts, faces, normals, values =  skimage.measure.marching_cubes(
        sdf_prediction.detach().cpu().numpy().reshape(100, 100, 100), level=-0.01, spacing=(0.01, 0.01, 0.01)
    )
    return verts, faces, normals, values

def extract_latent_code(dataset):
    """Extract latent code from the collected data.
    Returns:
        latent_code = torch.tensor of shape (1, dimension_latent)
    """
    samples_dict_path = os.path.join(os.path.dirname(results.__file__), f'samples_dict_{dataset}.npy')
    samples_dict = np.load(samples_dict_path, allow_pickle=True).item()
    obj_indices = list(samples_dict.keys())
    # test for one object
    obj_index = obj_indices[0]
    latent_code = torch.Tensor(samples_dict[obj_index]['latent_code'])
    return latent_code

def combine_latent_volum(latent, volum_grid):
    """Args:
        - latent: torch.Tensor of shape ([1, dimension_latent])
        - volum_grid: torch.Tensor of shape ([1000000, 3])
    """
    latent_full = torch.tile(latent, (volum_grid.shape[0], 1))   # repeat the latent code N times for stacking
    return torch.hstack((latent_full, volum_grid))

def generate_volum_grid():
    """Generate volumetric grid as a torch.Tensor, size([1000000, 3])"""
    x = torch.arange(-1, 1, 0.02)
    y = torch.arange(-1, 1, 0.02)
    z = torch.arange(-1, 1, 0.02)
    grid = torch.meshgrid(x, y, z)
    volum_grid = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)
    return volum_grid

def main(args):
    run_dir = os.path.join(os.path.dirname(runs.__file__), args.run_folder)
    weights_path = os.path.join(run_dir, 'weights.pt')
    model = sdf_model.SDFModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()
    # Latent code
    latent_code = extract_latent_code(args.dataset)
    # Volumetric grid as a torch.Tensor, size([1000000, 3])
    volum_grid = generate_volum_grid()
    # Combine volumetric grid and latent code
    data = combine_latent_volum(latent_code, volum_grid)
    # Predict
    with torch.no_grad():
        sdf_prediction = model(data)
    #mesh = extract_mesh(volum_grid, sdf_prediction)
    verts, faces, normals, values = extract_mesh_skimage(sdf_prediction)
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=None)
    trimesh.repair.fix_inversion(mesh, multibody=False)
    mesh_dict = dict()
    mesh_dict['verts'] = np.asarray(mesh.vertices)
    mesh_dict['faces'] = np.asarray(mesh.faces)
    mesh.show()
    np.save(run_dir, mesh_dict)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_folder", type=str
    )
    parser.add_argument(
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore' or 'PartNetMobility'"
    )
    args = parser.parse_args()
    main(args)