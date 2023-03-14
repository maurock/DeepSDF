import argparse
import numpy as np
import os
from results import runs_touch_sdf
from utils import utils_deepsdf
import plotly.graph_objects as go
import trimesh

def main(args):
    # Set directories
    infer_latent_dir = os.path.join(os.path.dirname(runs_touch_sdf.__file__), args.folder_touch_sdf, args.folder_infer_latent)
    touch_sdf_dir = os.path.join(os.path.dirname(runs_touch_sdf.__file__), args.folder_touch_sdf)

    # Load checkpoint
    checkpoint_path = os.path.join(infer_latent_dir, 'checkpoint_dict.npy')
    checkpoint_dict = np.load(checkpoint_path, allow_pickle=True).item()

    # Load original mesh and vertices
    original_mesh = trimesh.load(os.path.join(touch_sdf_dir, 'mesh_deepsdf.obj'))
    original_vertices = original_mesh.vertices

    for num_sample in checkpoint_dict.keys():
        # Read data
        vertices, faces = checkpoint_dict[num_sample]['mesh']
        pointclouds_deepsdf = checkpoint_dict[num_sample]['pointcloud']
        signed_distance = checkpoint_dict[num_sample]['sdf'].numpy().ravel()

        # Save mesh
        mesh_path = os.path.join(infer_latent_dir, str(num_sample), f'final_mesh.html')
        utils_deepsdf.save_meshplot(vertices, faces, mesh_path)

        # Aave mesh as obj
        obj_path = os.path.join(infer_latent_dir, str(num_sample), f"final_mesh.obj")
        trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type='obj')

        # Save pointclouds
        fig = go.Figure([
            go.Scatter3d(x=original_vertices[:, 0], y=original_vertices[:, 1],z=original_vertices[:, 2], 
            mode='markers', marker=dict(size=1)),
            go.Scatter3d(x=pointclouds_deepsdf[:, 0], y=pointclouds_deepsdf[:, 1],z=pointclouds_deepsdf[:, 2], 
            mode='markers', marker=dict(size=2, color=signed_distance.ravel(), 
                                cmin=np.amin(signed_distance), cmax=np.amax(signed_distance), colorscale='rdbu',
                                showscale=True))
            ]         
        )
        fig.write_html(os.path.join(infer_latent_dir, str(num_sample), f'final_pointclouds.html'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_touch_sdf", default=0, type=str, help="Folder containing the checkpoint of the touch_sdf test, e.g. 21_02_134102"
    )
    parser.add_argument(
        "--folder_infer_latent", default=0, type=str, help="Folder containing data related to the latent code inference, e.g infer_latent_22_02_111638"
    )
    args = parser.parse_args()

    main(args)