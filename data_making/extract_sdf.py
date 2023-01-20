import numpy as np
import results 
import os
from utils import utils_deepsdf, utils_mesh
import argparse
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
from mesh_to_sdf.surface_point_cloud import *
from tqdm import tqdm 
import point_cloud_utils as pcu
import data.ShapeNetCoreV2urdf as ShapeNetCoreV2
from glob import glob
import pybullet as pb

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
    
    if args.method == 'mesh_to_sdf':
        # This works with the PartNet-Mobility Dataset. It requires results/objs_dict.npy containing objects for each class.
        # This dictionary is created by extract_urdf.py.
        if not os.path.exists(os.path.join(os.path.dirname(results.__file__), f'objs_dict_{args.dataset}.npy')):
            print('Object dictionary does not exist. Please create it by running extract_urdf.py')
            exit()

        objs_dict = np.load(os.path.join(os.path.dirname(results.__file__), f'objs_dict_{args.dataset}.npy'), allow_pickle=True).item()

        samples_dict = dict()

        for obj_idx in tqdm(list(objs_dict.keys())[1:30]):

            samples_dict[obj_idx] = dict()

            mesh = trimesh.Trimesh(objs_dict[obj_idx]['verts'], objs_dict[obj_idx]['faces'])

            points, sdf = sample_sdf_near_surface(mesh, number_of_points=args.num_samples_total, sign_method=args.sign_method, auto_scaling=False, scale_ratio = 1.414)

            # TODO: temporary solution. Sample points and only keep the
            # negative ones to reduce the ratio positive/negative 
            neg_idxs = np.where(sdf < 0)[0]
            pos_idxs = np.where(sdf > 0)[0]
            ratio = float(len(pos_idxs))/float(len(neg_idxs))

            while ratio > 2:

                points_temp, sdf_temp = sample_sdf_near_surface(mesh, number_of_points=args.num_samples_total, sign_method=args.sign_method, auto_scaling=False, scale_ratio = 1.414)

                points = np.vstack((points, points_temp[sdf_temp<0]))
                sdf = np.hstack((sdf, sdf_temp[sdf_temp < 0]))

                neg_idxs = np.where(sdf < 0)[0]
                pos_idxs = np.where(sdf > 0)[0]
                ratio = float(len(pos_idxs))/float(len(neg_idxs))

                print(f'ratio: {ratio}')

            samples_dict[obj_idx]['samples'] = points
            samples_dict[obj_idx]['sdf'] = sdf
            samples_dict[obj_idx]['latent_class'] = np.array([obj_idx], dtype=np.int32)
            samples_dict[obj_idx]['samples_latent_class'] = combine_sample_latent(samples_dict[obj_idx]['samples'], samples_dict[obj_idx]['latent_class'])

    elif args.method == 'pcu':
        
        # Full paths to all .obj
        obj_paths = glob(os.path.join(os.path.dirname(ShapeNetCoreV2.__file__), '*', '*', '*.obj'))

        samples_dict = dict()
        
        # Store conversion between object index (int) and its folder name (str)
        idx_str2int_dict = dict()
        idx_int2str_dict = dict()

        for obj_idx, obj_path in enumerate(obj_paths):

            # Object unique index. Str to int by byte encoding
            obj_idx_str = obj_path.split(os.sep)[-2]
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

            samples_dict[obj_idx]['samples'] = p_total
            samples_dict[obj_idx]['sdf'] = sdf
            samples_dict[obj_idx]['latent_class'] = np.array([obj_idx], dtype=np.int32)
            samples_dict[obj_idx]['samples_latent_class'] = combine_sample_latent(samples_dict[obj_idx]['samples'], samples_dict[obj_idx]['latent_class'])
    else: 
        print('Choose a valid method')
        exit()      

    #_debug_plot(samples_dict[obj_idx])  
    np.save(os.path.join(os.path.dirname(results.__file__), f'samples_dict_{args.dataset}.npy'), samples_dict)

    np.save(os.path.join(os.path.dirname(results.__file__), 'idx_int2str_dict.npy'), idx_str2int_dict)
    np.save(os.path.join(os.path.dirname(results.__file__), 'idx_str2int_dict.npy'), idx_int2str_dict)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore' or 'PartNetMobility'"
    )
    parser.add_argument(
        '--method', default='pcu', type=str, help="Choose between 'mesh_to_sdf' or 'pcu'. 'mesh_to_sdf' uses the python package " +
                                                    "mesh_to_sdf, 'pcu' uses the python package point-cloud-utils"
    )
    # Args for mesh_to_sdf
    parser.add_argument(
        '--num_samples_total', default=5000, type=int, help="Num of total samples, mesh_to_sdf method"
    )
    parser.add_argument(
        '--sign_method', default='depth', type=str, help="Mode to determine the sign of the SDF, mesh_to_sdf method"
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