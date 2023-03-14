import pybullet as p
import pybullet_utils.bullet_client as bc
import numpy as np
import os
from cri_robot_arm import CRIRobotArm
from tactile_gym.assets import add_assets_path
from utils import utils_sample, utils_mesh, utils_deepsdf, utils_raycasting
import argparse
import data.ShapeNetCoreV2urdf as ShapeNetCore
from model import model_sdf, model_touch
import torch
import data
from results import runs_touch, runs_sdf, runs_touch_sdf
import trimesh
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
import point_cloud_utils as pcu
import results
import matplotlib.pyplot as plt
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Second step of the pipeline: predict the object shape from the touch data"""
#@profile
def main(args):
    # Logging
    test_dir = os.path.join(os.path.dirname(runs_touch_sdf.__file__), args.folder_touch_sdf, f"infer_latent_{datetime.now().strftime('%d_%m_%H%M%S')}")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    log_path = os.path.join(test_dir, 'settings.txt')
    args_dict = vars(args)  # convert args to dict to write them as json
    with open(log_path, mode='a') as log:
        log.write('Settings:\n')
        log.write(json.dumps(args_dict).replace(', ', ',\n'))
        log.write('\n\n')

    # Load sdf model
    sdf_model = model_sdf.SDFModelMulti(num_layers=8, no_skip_connections=False).float().to(device)
    
    # Load weights for sdf model
    weights_path = os.path.join(os.path.dirname(runs_sdf.__file__), args.folder_sdf, 'weights.pt')
    sdf_model.load_state_dict(torch.load(weights_path, map_location=device))
    sdf_model.eval()
    
    # Initial verts of the default touch chart
    chart_location = os.path.join(os.path.dirname(data.__file__), 'touch_chart.obj')
    initial_verts, initial_faces = utils_mesh.load_mesh_touch(chart_location)
    initial_verts = torch.unsqueeze(initial_verts, 0)

    # Instantiate grid coordinates for mesh extraction
    coords, grid_size_axis = utils_deepsdf.get_volume_coords(args.resolution)
    coords = coords.clone().to(device)
    coords_batches = torch.split(coords, 500000)

    # Save checkpoint
    checkpoint_dict = dict()
    checkpoint_path = os.path.join(test_dir, 'checkpoint_dict.npy')

    data_folders = glob(os.path.join(os.path.dirname(runs_touch_sdf.__file__), args.folder_touch_sdf, 'data', '*/'))

    # Infer latent code
    for data_folder in data_folders:

        # Sample folder to store tensorboard log and inferred latent code
        num_sample = data_folder.split('/')[-2]

        # If we don't want to reconstruct all, only reconstruct the object with the specified number of samples
        if (args.mode_reconstruct == 'fixed') and (int(num_sample) != (int(args.num_samples) - 1)):
            continue

        sample_dir = os.path.join(test_dir, num_sample)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        writer = SummaryWriter(log_dir=sample_dir)

        # Load pointclouds and sdf ground truth
        points_sdf = torch.load(os.path.join(data_folder, 'points_sdf.pt'), map_location=device)
        pointclouds_deepsdf = points_sdf[0]
        sdf_gt = points_sdf[1]

        # Infer latent code
        best_latent_code = sdf_model.infer_latent_code(args, pointclouds_deepsdf, sdf_gt, writer)

        # Extract mesh obtained with the latent code optimised at inference
        sdf = utils_deepsdf.predict_sdf(best_latent_code, coords_batches, sdf_model)
        vertices_deepsdf, faces_deepsdf = utils_deepsdf.extract_mesh(grid_size_axis, sdf)

        # Save mesh, pointclouds, and their signed distance
        checkpoint_dict[num_sample] = dict()
        checkpoint_dict[num_sample]['mesh'] = [vertices_deepsdf, faces_deepsdf]
        checkpoint_dict[num_sample]['pointcloud'] = pointclouds_deepsdf.cpu()
        checkpoint_dict[num_sample]['sdf'] = sdf_gt.cpu()
        checkpoint_dict[num_sample]['latent_code'] = best_latent_code.cpu()
        np.save(checkpoint_path, checkpoint_dict)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Argument for inference
    parser.add_argument(
        "--folder_touch_sdf", default=0, type=str, help="Folder containing the collected point clouds"
    )
    parser.add_argument(
        "--folder_sdf", default=0, type=str, help="Folder containing the sdf model weights"
    )
    parser.add_argument(
        "--latent_size", default=128, type=int, help="Folder containing the touch model weights"
    )
    parser.add_argument(
        "--optimiser", default='Adam', type=str, help="Choose the optimiser out of [Adam, LBFGS]"
    )
    parser.add_argument(
        "--lr", default=0.00001, type=float, help="Learning rate to infer the latent code"
    )
    parser.add_argument(
        "--lr_scheduler", default=False, action='store_true', help="Learning rate to infer the latent code"
    )
    parser.add_argument(
        "--sigma_regulariser", default=0.01, type=float, help="Regulariser for the loss function"
    )
    parser.add_argument(
        "--lr_multiplier", default=0.5, type=float, help="Learning rate multiplier for the scheduler"
    )
    parser.add_argument(
        "--patience", default=50, type=float, help="Patience for latent code inference"
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs for latent code inference"
    )
    parser.add_argument(
        "--clamp", default=False, action='store_true', help="Clip the network prediction"
    )
    parser.add_argument(
        "--clamp_value", type=float, default=0.1, help="Value of the clip"
    )
    parser.add_argument(
        "--langevin_noise", type=float, default=0, help="If this value is higher than 0, it adds noise to the latent space after every update."
    )
    parser.add_argument(
        "--resolution", type=int, default=50, help="Resolution of the extracted mesh"
    )
    parser.add_argument(
        "--mode_reconstruct", default='all', type=str, help="Choose between 'all' or 'fixed' to choose between reconstructing for all the collected samples, or only for the specified number of samples. E.g. if args.num_samples=10 and args.mode_reconstruct='fixed', then only the file with 10 samples will be used to reconstruct the object. Otherwise, all the files will be used, therefore the script reconstruct up to 10 samples."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samplings on the objects"
    )
    args = parser.parse_args()

    # args.folder_sdf ='23_01_095414'
    # args.folder_touch_sdf ='21_02_134102' 
    # args.lr_scheduler = True
    # args.epochs =5 
    # args.lr =0.00005 
    # args.patience =100 
    # args.resolution =20 

    main(args)