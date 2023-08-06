import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import pkgutil
import os
from cri_robot_arm import CRIRobotArm
from tactile_gym.assets import add_assets_path
from utils import utils_sample, utils_mesh, utils_deepsdf, utils_raycasting
import argparse
import data.ShapeNetCoreV2urdf as ShapeNetCore
import data.ABCurdf as ABC
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
import scripts.pipeline_tactile as pipeline_tactile
import scripts.pipeline_deepsdf as pipeline_deepsdf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Demo to reconstruct objects using tactile-gym.
"""
def main(args):

    # Pipeline to collect touch data
    args.folder_touch_sdf = pipeline_tactile.main(args)
    
    # Pipeline to reconstruct object from touch data
    pipeline_deepsdf.main(args)

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # Arguments for sampling
    parser.add_argument(
        "--show_gui", default=False, action='store_true', help="Show PyBullet GUI"
    )
    parser.add_argument(
        "--show_tactile", default=False, action='store_true', help="Show tactile image"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samplings on the objects"
    )
    parser.add_argument(
        "--render_scene", default=False, action='store_true', help="Render scene at touch"
    )
    parser.add_argument(
        "--scale", default=0.2, type=float, help="Scale of the object in simulation wrt the urdf object"
    )
    parser.add_argument(
        "--folder_touch", default=0, type=str, help="Folder containing the touch model weights"
    )
    parser.add_argument(
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore', 'ShapeNetCore_test', or 'ABC'"
    )
    parser.add_argument(
        "--augment_points_std", default=0.002, type=float, help="Standard deviation of the Gaussian used to sample points along normals (if augment_points is True)"
    )
    parser.add_argument(
        "--augment_points_num", default=5, type=int, help="Number of points to sample along normals"
    )
    parser.add_argument(
        "--augment_multiplier_out", default=1, type=int, help="multiplier to augment the positive distances"
    )
    parser.add_argument(
        "--obj_folder", type=str, default='', help="Object to reconstruct as obj_class/obj_category, e.g. 02818832/1aa55867200ea789465e08d496c0420f"
    )

    # Arguments for deepsdf
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
        "--no_mesh_extraction", default=False, action='store_true', help="When true, do not extract the resulting mesh as html and obj, as well as the touches point cloud."
    )
    parser.add_argument(
        "--num_samples_extraction", type=int, default=10, nargs='+', help="Number of samples on the objects. It can be a single number or a list of numbers, e.g. 10 20 30."
    )
    parser.add_argument(
        "--inner_dim", type=int, default=512, help="Inner dimensions of the network"
    )
    parser.add_argument(
        "--positional_encoding_embeddings", type=int, default=0, help="Number of embeddingsto use for positional encoding. If 0, no positional encoding is used."
    )
    args = parser.parse_args()

    main(args)