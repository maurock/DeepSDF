import numpy as np
import results 
import os
from utils import utils_mesh
import argparse
import plotly.graph_objects as go
import point_cloud_utils as pcu
import data.ShapeNetCoreV2urdf as ShapeNetCoreV2
import data.ABC_train_urdf as ABC_train
from glob import glob
import pybullet as pb
from datetime import datetime

"""
For each object, sample points and store their distance to the nearest triangle.
Sampling follows the approach used in the DeepSDF paper.
"""

def combine_sample_latent(samples, latent_class):
    """Combine each sample (x, y, z) with the latent code generated for this object.
    Args:
        samples: collected points, np.array of shape (N, 3)
        latent: randomly generated latent code, np.array of shape (1, args.latent_size)
    Returns:
        combined hstacked latent code and samples, np.array of shape (N, args.latent_size + 3)
    """
    latent_class_full = np.tile(latent_class, (samples.shape[0], 1))   # repeat the latent code N times for stacking
    return np.hstack((latent_class_full, samples))

def _debug_plot(samples, dist=True):
    points = samples['samples']
    sdfs = samples['sdf']
    inner = points[sdfs>0]
    outer = points[sdfs<0]

    fig = go.Figure(
        [
            go.Scatter3d(x=inner[:, 0], y=inner[:, 1],z=inner[:, 2], mode='markers', marker=dict(size=2)),
            go.Scatter3d(x=outer[:, 0], y=outer[:, 1],z=outer[:, 2], mode='markers', marker=dict(size=2))
        ]
    )
    fig.show()

def main(args):

    timestamp_run = datetime.now().strftime('_%d%m')   # timestamp to use for logging data
    
    # Full paths to all .obj
    dataset_module = ShapeNetCoreV2 if args.dataset == 'ShapeNetCore' else ABC_train if args.dataset == 'ABC' else exit('Dataset not supported')
    print(f'Using dataset {dataset_module.__name__}')

    obj_paths = glob(os.path.join(os.path.dirname(dataset_module.__file__), '*', '*', '*.obj'))

    samples_dict = dict()        
    # Store conversion between object index (int) and its folder name (str)
    idx_str2int_dict = dict()
    idx_int2str_dict = dict()

    for obj_idx, obj_path in enumerate(obj_paths):

        # Object unique index. Str to int by byte encoding
        obj_idx_str = os.sep.join(obj_path.split(os.sep)[-3:-1]) # e.g. '02958343/1a2b3c4d5e6f7g8h9i0j'
        idx_str2int_dict[obj_idx_str] = obj_idx
        idx_int2str_dict[obj_idx] = obj_idx_str

        # Dictionary to store the samples and SDFs
        samples_dict[obj_idx] = dict()

        try:
            verts, faces = pcu.load_mesh_vf(obj_path)
        except:
            continue
        
        # In Shapenet, the front is the -Z axis with +Y still being the up axis. 
        # Rotate objects to align with the canonical axis. 
        rot_M = pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler([np.pi/2, 0, -np.pi/2]))
        rot_M = np.array(rot_M).reshape(3, 3)
        verts = utils_mesh.rotate_pointcloud(verts, [np.pi/2, 0, -np.pi/2])

        # Generate random points in the predefined volume that surrounds all the shapes.
        # NOTE: ShapeNet shapes are normalized within [-1, 1]^3
        p_vol = np.random.rand(args.num_samples_in_volume, 3) * 2 - 1

        # Sample within the object's bounding box. This ensures a higher ratio between points inside and outside the surface.
        v_min, v_max = verts.min(0), verts.max(0)
        p_bbox = np.random.uniform(low=[v_min[0], v_min[1], v_min[2]], high=[v_max[0], v_max[1], v_max[2]], size=(args.num_samples_in_bbox, 3))

        # Sample points on the surface as face ids and barycentric coordinates
        fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, args.num_samples_on_surface)

        # Compute 3D coordinates and normals of surface samples
        p_surf = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)

        p_total = np.vstack((p_vol, p_bbox, p_surf))

        # Comput the SDF of the random points
        sdf, _, _  = pcu.signed_distance_to_mesh(p_total, verts, faces)

        samples_dict[obj_idx]['sdf'] = sdf
        # The samples are p_total, while the latent class is [obj_idx]
        samples_dict[obj_idx]['samples_latent_class'] = combine_sample_latent(p_total, np.array([obj_idx], dtype=np.int32))

        # Save the samples and SDFs at regular intervals
        if obj_idx % 100 == 0:
            np.save(os.path.join(os.path.dirname(results.__file__), f'samples_dict_{args.dataset}_{timestamp_run}.npy'), samples_dict)  

    #_debug_plot(samples_dict[obj_idx])  
    np.save(os.path.join(os.path.dirname(results.__file__), f'samples_dict_{args.dataset}_{timestamp_run}.npy'), samples_dict)

    np.save(os.path.join(os.path.dirname(results.__file__), f'idx_str2int_dict_{args.dataset}_{timestamp_run}.npy'), idx_str2int_dict)
    np.save(os.path.join(os.path.dirname(results.__file__), f'idx_int2str_dict_{args.dataset}_{timestamp_run}.npy'), idx_int2str_dict)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore' or 'ABC'"
    )
    # Point-cloud-utils
    parser.add_argument(
        '--num_samples_on_surface', default=5000, type=int, help="Num samples on object surface"
    )  
    parser.add_argument(
        '--num_samples_in_bbox', default=5000, type=int, help="Num samples within the object bounding box"
    )  
    parser.add_argument(
        '--num_samples_in_volume', default=1000, type=int, help="Num samples within the predefined volume"
    )  
    args = parser.parse_args()
    
    main(args)