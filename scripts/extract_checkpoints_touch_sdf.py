import argparse
import numpy as np
import os
from results import runs_touch_sdf
from utils import utils_deepsdf
import plotly.graph_objects as go

def main(args):
    test_dir = os.path.join(os.path.dirname(runs_touch_sdf.__file__), args.folder_touch_sdf)
    checkpoint_path = os.path.join(test_dir, 'checkpoint_dict.npy')
    checkpoint_dict = np.load(checkpoint_path, allow_pickle=True).item()
    
    for num_sample in checkpoint_dict.keys():
        # Read data
        vertices, faces = checkpoint_dict[num_sample]['mesh']
        original_mesh, pointclouds_deepsdf = checkpoint_dict[num_sample]['pointcloud']
        signed_distance = checkpoint_dict[num_sample]['sdf'].numpy().ravel()

        # Save mesh
        mesh_path = os.path.join(test_dir, str(num_sample), f'final_mesh.html')
        utils_deepsdf.save_meshplot(vertices, faces, mesh_path)

        # Save pointclouds
        fig = go.Figure([
            go.Scatter3d(x=original_mesh[:, 0], y=original_mesh[:, 1],z=original_mesh[:, 2], 
            mode='markers', marker=dict(size=2)),
            go.Scatter3d(x=pointclouds_deepsdf[:, 0], y=pointclouds_deepsdf[:, 1],z=pointclouds_deepsdf[:, 2], 
            mode='markers', marker=dict(size=2, color=signed_distance.ravel(), 
                                cmin=np.amin(signed_distance), cmax=np.amax(signed_distance), colorscale='rdbu',
                                showscale=True))
            ]         
        )
        fig.write_html(os.path.join(test_dir, str(num_sample), f'final_pointclouds.html'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_touch_sdf", default=0, type=str, help="Folder containing the checkpoint of the touch_sdf test"
    )
    args = parser.parse_args()

    main(args)