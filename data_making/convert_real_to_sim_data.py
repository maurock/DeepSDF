"""This script converts data collecte din real life with the format required by the pipeline_deepsdf.py script. """
import numpy as np
import os
from utils import utils_mesh, utils_sample
import argparse
import torch
import data
from results import runs_touch_sdf
from datetime import datetime
import json
from glob import glob
import data.ShapeNetCoreV2urdf as ShapeNetCore
import trimesh 
import pybullet as pb
from data import real_objects
from PIL import Image
from model import model_touch
from results import runs_touch
import point_cloud_utils as pcu

"""Extracting the data provided by the real robot and creates a folder in results/runs_touch_sdf with the same format as the data collected in simulation. """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # Logging
    test_dir = os.path.join(os.path.dirname(runs_touch_sdf.__file__), datetime.now().strftime('%d_%m_%H%M%S'))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    log_path = os.path.join(test_dir, 'settings.txt')
    args_dict = vars(args)  # convert args to dict to write them as json
    with open(log_path, mode='a') as log:
        log.write('Settings:\n')
        log.write(json.dumps(args_dict).replace(', ', ',\n'))
        log.write('\n\n')

    # Set initial object pose
    initial_obj_rpy = [np.pi/2, 0, -np.pi/2]
    initial_obj_orn = pb.getQuaternionFromEuler(initial_obj_rpy)
    initial_obj_pos = [0.5, 0.0, 0]

    # Load DeepSDF object
    obj_dir = os.path.join(os.path.dirname(ShapeNetCore.__file__), args.obj_folder)
    # Load object and get world frame coordinates.
    obj_path = os.path.join(obj_dir, "model.obj")
    mesh_original = utils_mesh._as_mesh(trimesh.load(obj_path))
    # Process object vertices to match the real object
    vertices_wrld = utils_mesh.rotate_pointcloud(mesh_original.vertices, initial_obj_rpy) * args.scale + initial_obj_pos
    mesh = trimesh.Trimesh(vertices=vertices_wrld, faces=mesh_original.faces)

    # Store processed mesh in deepsdf pose
    verts_deepsdf = utils_mesh.rotate_pointcloud(mesh_original.vertices, initial_obj_rpy)
    trimesh.Trimesh(vertices=verts_deepsdf, faces=mesh_original.faces).export(os.path.join(test_dir, 'mesh_deepsdf.obj'))

    # Set folders and paths for poses and images
    real_obj_folder = os.path.join(os.path.dirname(real_objects.__file__), args.folder_real_data)
    poses_path = os.path.join(real_obj_folder, 'poses_world.npy')
    images_dir = os.path.join(real_obj_folder, 'generated_sim_imgs')

    def get_number_from_path(path):
        return int(path.split('_')[-1].split('.')[0])

    images_paths = sorted(glob(os.path.join(images_dir, 'sim_img_*.png')), key=get_number_from_path)

    # Load initial touch chart
    chart_location = os.path.join(os.path.dirname(data.__file__), 'touch_chart.obj')
    initial_verts, initial_faces = utils_mesh.load_mesh_touch(chart_location)
    initial_verts = torch.unsqueeze(initial_verts, 0)

    # Load data
    poses = np.load(poses_path)
    cameras = [np.array(Image.open(image_path).getdata()) for image_path in images_paths]
    images = [torch.tensor(Image.open(image_path).getdata(), requires_grad=False, dtype=torch.float32)/255 for image_path in images_paths]
    images = [tensor.view(1, 1, 256, 256) for tensor in images]

    # Load models
    # Load touch model
    touch_model = model_touch.Encoder()
    weights_path = os.path.join(os.path.dirname(runs_touch.__file__),  args.folder_touch, 'weights.pt')
    touch_model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))

    # Total signed distance
    sdf_gt = torch.tensor([]).view(0, 1).to(device)

    # Filter out first datapoint from images and poses
    if args.discard_calibration_image:
        images.pop(0)
        cameras.pop(0)
        poses = poses[1:, :, :]   

    # Shuffle data
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = [images[i] for i in indices]
    cameras = [cameras[i] for i in indices]
    poses = poses[indices, :, :]    

    # Instantiate pointcloud for DeepSDF prediction
    pointclouds_deepsdf = torch.tensor([]).view(0, 3).to(device)

    ##############################################################################################
    # During the first stage, we collect tactile images, map them to point clouds, and store them.
    num_sample = 1
    for i in range(0, len(images)):
        
        # Check that the robot is touching the object and the avg colour pixel doesn't exceed a threshold
        tactile_img = images[i]
        
        # Predict vertices from tactile image (TCP frame, Sim scale)
        predicted_verts = touch_model(tactile_img, initial_verts)[0]

        # Create mesh from predicted vertices
        predicted_mesh = trimesh.Trimesh(predicted_verts.detach().cpu(), initial_faces.cpu())

        # Sample pointcloud from predicted mesh in Sim coordinates (TCP frame, Sim scale)
        predicted_pointcloud = utils_mesh.mesh_to_pointcloud(predicted_mesh, 500)

         # Translate to world frame
        pos_wrld = np.array([poses[i, 0, :]])
        rot_Q_wrld = pb.getQuaternionFromEuler(poses[i, 1, :])
        rot_M_wrld = np.array(pb.getMatrixFromQuaternion(rot_Q_wrld)).reshape(1, 3, 3)
        predicted_pointcloud_wrld = utils_mesh.translate_rotate_mesh(pos_wrld, rot_M_wrld, predicted_pointcloud[None, :, :], initial_obj_pos)

        # Rescale from Sim scale to DeepSDF scale
        pointcloud_deepsdf_np = (predicted_pointcloud_wrld / args.scale)[0]  # shape (n, 3)

        # Concatenate predicted pointclouds of the touch charts from all samples
        pointcloud_deepsdf = torch.from_numpy(pointcloud_deepsdf_np).float().to(device)  # shape (n, 3)
        pointclouds_deepsdf = torch.vstack((pointclouds_deepsdf, pointcloud_deepsdf))   
        
        # The sdf of points on the object surface is 0.
        sdf_gt = torch.vstack((sdf_gt, torch.zeros(size=(pointcloud_deepsdf.shape[0], 1)).to(device)))

        # Add randomly sampled points from normals
        if args.augment_points_num > 0:
            # The sensor direction is given by the vectors pointing from the pointcloud to the TCP position
            # We need to first convert the TCP position to DeepSDF scale
            rpy_wrld = np.array(poses[i, 1, :]) # TCP orientation
            normal_wrk = np.array([[0, 0, 1]])
            normal_wrld = utils_mesh.rotate_pointcloud(normal_wrk, rpy_wrld)[0]
            TCP_pos_deepsdf = (pos_wrld -  0.01 * normal_wrld)  # move it slightly away from the object
            TCP_pos_deepsdf = (TCP_pos_deepsdf -  initial_obj_pos)/ args.scale # convert to DeepSDF scale
            sensor_dirs = TCP_pos_deepsdf - pointcloud_deepsdf_np
            
            # Estimate point cloud normals
            _, n = pcu.estimate_point_cloud_normals_knn(pointcloud_deepsdf_np, 32, view_directions=sensor_dirs)

            # Sample along normals and return points and distances
            pointcloud_along_norm_np, signed_distance_np = utils_sample.sample_along_normals(
                std_dev=args.augment_points_std, pointcloud=pointcloud_deepsdf_np, normals=n, N=args.augment_points_num, augment_multiplier_out=args.augment_multiplier_out)
            pointcloud_along_norm = torch.from_numpy(pointcloud_along_norm_np).float().to(device)
            sdf_normal_gt = torch.from_numpy(signed_distance_np).float().to(device)

            pointclouds_deepsdf = torch.vstack((pointclouds_deepsdf, pointcloud_along_norm))
            sdf_gt = torch.vstack((sdf_gt, sdf_normal_gt))

        # Save pointclouds
        points_sdf = [pointclouds_deepsdf.detach().cpu(), sdf_gt.detach().cpu()]
        points_sdf_dir = os.path.join(test_dir, 'data', str(num_sample))
        if not os.path.isdir(points_sdf_dir):
            os.makedirs(points_sdf_dir)
        torch.save(points_sdf, os.path.join(points_sdf_dir, f'points_sdf.pt'))

        # Store tactile image
        np.save(os.path.join(points_sdf_dir, f'tactile_img.npy'), cameras[i])

        num_sample+=1

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Argument for inference
    parser.add_argument(
        "--folder_real_data", default=0, type=str, help="Folder containing the collected real data"
    )
    parser.add_argument(
        "--obj_folder", type=str, default='', help="Object to reconstruct as obj_class/obj_category, e.g. 02818832/1aa55867200ea789465e08d496c0420f"
    )
    parser.add_argument(
        "--folder_touch", default=0, type=str, help="Folder containing the touch model weights"
    )
    parser.add_argument(
        "--scale", default=0.2, type=float, help="Scale of the object in simulation wrt the urdf object"
    )
    parser.add_argument(
        "--augment_points_std", default=0.002, type=float, help="Standard deviation of the Gaussian used to sample points along normals (if augment_points is True)"
    )
    parser.add_argument(
        "--augment_points_num", default=0, type=int, help="Number of points to sample along normals (if augment_points is True)"
    )
    parser.add_argument(
        "--discard_calibration_image", action='store_true',  help="Discard the first image and pose, which is usually used for calibration"
    )
    parser.add_argument(
        "--augment_multiplier_out", default=1, type=int, help="multiplier to augment the positive distances"
    )

    args = parser.parse_args()  

    # args.folder_real_data = 'jar'
    # args.obj_folder = '03797390/ff1a44e1c1785d618bca309f2c51966a'
    # args.folder_touch = '30_05_1633' 
    # args.scale = 0.2
    # args.discard_calibration_image = True
    # args.augment_points_num = 5
    # args.augment_points_std = 0.005
    # args.augment_multiplier_out = 5
    main(args)