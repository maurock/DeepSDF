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

"""
For each object, sample points and store their distance to the nearest triangle.
Sampling follows the approach used in the DeepSDF paper.
"""

def compute_sdf(verts, faces, samples):
    mesh = trimesh.Trimesh(verts, faces)
    trimesh.repair.fix_inversion(mesh)
    proximity = trimesh.proximity.ProximityQuery(mesh)
    sdfs = proximity.signed_distance(samples)
    return sdfs

def sample_far_surface(args):
    """
    Sampling uniformly within a unit sphere
    """
    phi = np.random.uniform(0, 2 * np.pi, size=args.num_samples_far_surface)
    costheta = np.random.uniform(-1, 1, size=args.num_samples_far_surface)
    u = np.random.uniform(0, 1, size=args.num_samples_far_surface)
    theta = np.arccos(costheta)
    r = 1.1 * np.cbrt(u)    # R = 1.003 as presented in the original paper
    x = r * np.sin(theta) * np.cos( phi )
    y = r * np.sin(theta) * np.sin( phi )
    z = r * np.cos(theta)
    samples = np.vstack((x, y, z)).transpose(1, 0)
    return samples

def sample_near_surface(samples_on_surface):
    """Sampling near the surface using gaussian pdfs, variance=0.0025 and variance=0.00025
    as described in the paper"""
    var1 = 0.0025 
    var2 = 0.00025
    samples_var1 = np.random.normal(0, var1, size=samples_on_surface.shape[0] * samples_on_surface.shape[1])
    samples_var2 = np.random.normal(0, var2, size=samples_on_surface.shape[0] * samples_on_surface.shape[1])
    samples_near_surface_var1 = samples_on_surface + samples_var1.reshape(samples_on_surface.shape[0], samples_on_surface.shape[1])
    samples_near_surface_var2 = samples_on_surface + samples_var2.reshape(samples_on_surface.shape[0], samples_on_surface.shape[1])  
    samples_near_surface = np.vstack((samples_near_surface_var1, samples_near_surface_var2))
    return samples_near_surface

def sample_on_surface(obj_dict, args):
    verts = obj_dict['verts']
    faces = obj_dict['faces']
    mesh = trimesh.Trimesh(verts, faces)
    samples = utils_mesh.mesh_to_pointcloud(mesh, args.num_samples_on_surface)
    return samples

def sample(obj_dict, args):
    samples = np.array([]).reshape(0, 3)
    samples_on_surface = sample_on_surface(obj_dict, args)
    samples_near_surface = sample_near_surface(samples_on_surface)
    samples_far_surface = sample_far_surface(args)
    samples = np.vstack((samples_on_surface, samples_near_surface, samples_far_surface))
    return samples

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

    # load objs_dict from results/objs_dict.npy
    if not os.path.exists(os.path.join(os.path.dirname(results.__file__), f'objs_dict_{args.dataset}.npy')):
        print('Object dictionary does not exist. Please create it by running extract_urdf.py')
        exit()

    objs_dict = np.load(os.path.join(os.path.dirname(results.__file__), f'objs_dict_{args.dataset}.npy'), allow_pickle=True).item()

    samples_dict = dict()

    for obj_idx in tqdm(list(objs_dict.keys())[1:30]):

        samples_dict[obj_idx] = dict()

        if args.method == 'custom':
            samples_dict[obj_idx]['samples'] = sample(objs_dict[obj_idx], args)
            samples_dict[obj_idx]['sdf'] = compute_sdf(objs_dict[obj_idx]['verts'], objs_dict[obj_idx]['faces'], samples_dict[obj_idx]['samples'])

        elif args.method == 'library':

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

        else: 
            print('Choose a valid method')
            exit()        

        samples_dict[obj_idx]['latent_class'] = np.array([obj_idx], dtype=np.int32)

        samples_dict[obj_idx]['samples_latent_class'] = combine_sample_latent(samples_dict[obj_idx]['samples'], samples_dict[obj_idx]['latent_class'])

        #_debug_plot(samples_dict[obj_idx])  
    np.save(os.path.join(os.path.dirname(results.__file__), 'samples_dict.npy'), samples_dict)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_samples_on_surface', default=5000, type=int, help="Num samples on object surface, custom method"
    )  
    parser.add_argument(
        '--num_samples_far_surface', default=5000, type=int, help="Num samples far from the object surface, custom method"
    )    
    parser.add_argument(
        '--method', default='library', type=str, help="Choose between 'custom' or 'library'. 'Custom' samples and computes the SDF using" +
                                                    "a custom method. 'library' uses the mesh_to_sdf library."
    )
    parser.add_argument(
        '--num_samples_total', default=5000, type=int, help="Num of total samples, library method"
    )
    parser.add_argument(
        '--sign_method', default='depth', type=str, help="Mode to determine the sign of the SDF, library method"
    )
    parser.add_argument(
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore' or 'PartNetMobility'"
    )
    args = parser.parse_args()
    main(args)