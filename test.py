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
def extract_mesh(data, sdf):
    sdf_inside = data[(sdf.view(-1) < 0.01)]
    mesh = trimesh.voxel.ops.points_to_marching_cubes(sdf_inside.detach().cpu().numpy(), pitch=0.2)
    return mesh

def main(args):
    run_dir = os.path.join(os.path.dirname(runs.__file__), args.run_folder)
    weights_path = os.path.join(run_dir, 'weights.pt')
    sdf_model = model.SDFModel()
    sdf_model.load_state_dict(torch.load(weights_path))
    x = torch.arange(-1, 1, 0.01)
    y = torch.arange(-1, 1, 0.01)
    z = torch.arange(-1, 1, 0.01)
    data = torch.vstack((x, y, z)).transpose(1, 0)
    sdf = sdf_model(data)
    mesh = extract_mesh(data, sdf)
    mesh_dict = dict()
    mesh_dict['verts'] = np.asarray(mesh.vertices)
    mesh_dict['faces'] = np.asarray(mesh.faces)
    np.save(run_dir, mesh_dict)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_folder", type=str
    )
    args = parser.parse_args()
    main(args)