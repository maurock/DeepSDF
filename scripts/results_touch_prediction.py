"""Script to extract results of the touch prediction pipeline. It extracts the mesh,
the point cloud of the predicted touches, and computes their distance to the mesh"""
import numpy as np
import torch
import os
from glob import glob
from data import real_objects
from PIL import Image
from model import model_sdf, model_touch
from results import runs_touch, runs_sdf
from utils import utils_mesh, utils_deepsdf
import data
import plotly.graph_objects as go
import trimesh
import pybullet as pb
import cv2 
import matplotlib.pyplot as plt 
from model import model_sdf
from torch.utils.tensorboard import SummaryWriter
import argparse
import meshplot as mp
from data import ShapeNetCoreV2urdf
import point_cloud_utils as pcu 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # Extract required objects
    folder = os.path.join(os.path.dirname(real_objects.__file__), args.obj_folder)
    poses_path = os.path.join(folder, 'poses_world.npy')
    images_dir = os.path.join(folder, 'generated_sim_imgs')

    obj_path = dict()
    obj_path['camera'] = '02942699/6d036fd1c70e5a5849493d905c02fa86'
    obj_path['bowl'] = '02880940/1b4d7803a3298f8477bdcb8816a3fac9'
    obj_path['bottle'] = '02876657/3d758f0c11f01e3e3ddcd369aa279f39'
    obj_path['real_bottle'] = 'real_mesh/bottle'


    def get_number_from_path(path):
        return int(path.split('_')[-1].split('.')[0])

    images_paths = sorted(glob(os.path.join(images_dir, 'sim_img_*.png')), key=get_number_from_path)
    mesh_path = os.path.join(os.path.dirname(ShapeNetCoreV2urdf.__file__), obj_path[args.obj_category], 'model.obj')

    folder_touch = '30_05_1633' 
    chart_location = os.path.join(os.path.dirname(data.__file__), 'touch_chart.obj')
    initial_verts, initial_faces = utils_mesh.load_mesh_touch(chart_location)
    initial_verts = torch.unsqueeze(initial_verts, 0)

    # Load data
    poses = np.load(poses_path)
    images = [torch.tensor(Image.open(image_path).getdata(), requires_grad=False, dtype=torch.float32)/255 for image_path in images_paths]
    images = [tensor.view(1, 1, 256, 256) for tensor in images]

    # Load models
    touch_model = model_touch.Encoder()
    weights_path = os.path.join(os.path.dirname(runs_touch.__file__),  folder_touch, 'weights.pt')
    touch_model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))

    # Filter calibration image and set variables
    start_idx = 1 if args.discard_calibration_image else 0
    # Get rotations
    rotations_rad = poses[start_idx:, 1, :]
    # Positions world
    positions = poses[start_idx:, 0, :]
    # Images
    images = images[start_idx:]

    # Predict point clouds wrt TacTip frame
    pointclouds_wrk = []
    for image in images:
        # Predict point cloud
        with torch.no_grad():
            # Predict vertices in tcp workframe
            predicted_verts = touch_model(image, initial_verts)[0]
            # Extract local point cloud from predicted vertices
            touch_chart_mesh = trimesh.Trimesh(predicted_verts.numpy(), initial_faces)
            pointclouds_wrk.append(np.array(trimesh.sample.sample_surface(touch_chart_mesh, 1000)[0], dtype=np.float32))
            
    # Convert tactip frame to world frame
    pointclouds_wrld = []
    for idx, rot_rad in enumerate(rotations_rad):
        rot_Q_wrld = pb.getQuaternionFromEuler(rot_rad)
        # Translate to world frame
        pos_wrld = np.array([positions[idx]])
        rot_M_wrld = np.array(pb.getMatrixFromQuaternion(rot_Q_wrld)).reshape(1, 3, 3)
        pointcloud_wrld = utils_mesh.translate_rotate_mesh(pos_wrld, rot_M_wrld, pointclouds_wrk[idx][None, :, :], obj_initial_pos=[0,0,0])
        pointclouds_wrld.extend(pointcloud_wrld[0])
        
    # Pointcloud on object
    initial_obj_rpy = [np.pi/2, 0, -np.pi/2]
    initial_obj_pos = [0.5, 0.0, 0.0]

    # Load mesh from ShapeNetCoreV2
    v, f = pcu.load_mesh_vf(mesh_path)

    # Transform mesh coordinate by rotating and scaling the object
    v_transformed = utils_mesh.rotate_pointcloud(v, initial_obj_rpy) * 0.2 + initial_obj_pos

    # sdf is the signed distance for each query point
    sdf, _, _ = pcu.signed_distance_to_mesh(np.array(pointclouds_wrld), v_transformed, f)

    # MAE
    mae = np.mean(np.abs(sdf)) 

    print(f'MAE: {mae}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--obj_folder", type=str, default='', help="Object to reconstruct as obj_class/obj_category, e.g. 02818832/1aa55867200ea789465e08d496c0420f"
    )
    parser.add_argument(
        "--obj_category", type=str, default='', help="E.g. camera, bowl, bottle, real_bottle"
    )
    parser.add_argument(
        "--discard_calibration_image", action='store_true',  help="Discard the first image and pose, which is usually used for calibration"
    )
    args = parser.parse_args()

    main(args)