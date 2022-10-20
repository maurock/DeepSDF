import torch
import numpy
import results.runs as runs
import os
import model
import argparse
import trimesh
import numpy as np
"""
Test model
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def extract_mesh(data, sdf):
    sdf_inside = data[(sdf.view(-1)>-0.01) & (sdf.view(-1)<0.01)]
    mesh = trimesh.voxel.ops.points_to_marching_cubes(sdf_inside.detach().cpu().numpy(), pitch=0.02)
    return mesh

def main(args):
    run_dir = os.path.join(os.path.dirname(runs.__file__), args.run_folder)
    weights_path = os.path.join(run_dir, 'weights.pt')
    sdf_model = model.SDFModel().double().to(device)
    sdf_model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    sdf_model.eval()
    x = torch.arange(-1, 1, 0.02)
    y = torch.arange(-1, 1, 0.02)
    z = torch.arange(-1, 1, 0.02)
    grid = torch.meshgrid(x, y, z)
    data = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)
    with torch.no_grad():
        sdf = sdf_model(data)
    mesh = extract_mesh(data, sdf)
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
    args = parser.parse_args()

    main(args)