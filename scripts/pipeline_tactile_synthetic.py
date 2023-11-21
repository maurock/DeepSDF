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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_synthetic_touch(mesh, radius=0.05):
    """Given a mesh, it returns a point cloud of the mesh surface. It first samples points on the mesh surface and then
    it computes a radius around each point. The points of the mesh inside the radius are added to the point cloud.
    
    Params:
        - mesh: mesh of the object
        - patch_size: radius of the patch around a randomly selected point on the object surface
    
    Returns:
        - coords: points on the mesh surface
    """
    # sample points on the mesh surface
    points_np = np.array(trimesh.sample.sample_surface(mesh, 50000)[0], dtype=np.float32)
    # select a random element from points, which is an array of shape(n, 3)
    centre_np = points_np[np.random.randint(0, len(points_np))]

    centre = torch.from_numpy(centre_np).float().to(device)
    points = torch.from_numpy(points_np).float().to(device)

    # compute radius around each point
    coords = get_points_in_radius(points, centre, radius)

    return coords

def get_points_in_radius(points, centre, radius):
    """Given a point cloud, it returns the points inside a radius around each point in the point cloud.
    
    Params:
        - points: point cloud
        - centre: centre of the radius
        - radius: radius
    
    Returns:
        - coords: points inside the radius
    """
    dist = torch.norm(points - centre, dim=1)
    coords = points[dist < radius]

    return coords     

#@profile
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


    time_step = 1. / 960  # low for small objects

    if args.show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  


    else:
        pb = bc.BulletClient(connection_mode=p.DIRECT)
        egl = pkgutil.get_loader('eglRenderer')
        if (egl):
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")

    # Set physics engine parameters
    pb.setGravity(0, 0, -10)
    pb.setPhysicsEngineParameter(fixedTimeStep=time_step,
                                 numSolverIterations=150,  # 150 is good but slow
                                 numSubSteps=1,
                                 contactBreakingThreshold=0.0005,
                                 erp=0.05,
                                 contactERP=0.05,
                                 frictionERP=0.2,
                                 # need to enable friction anchors (maybe something to experiment with)
                                 solverResidualThreshold=1e-7,
                                 contactSlop=0.001,
                                 globalCFM=0.0001)        

    # Load the environment
    # plane_id = pb.loadURDF(
    #     add_assets_path("shared_assets/environment_objects/plane/plane.urdf")
    # )

   
    # Set initial object pose
    initial_obj_rpy = [np.pi/2, 0, -np.pi/2]
    initial_obj_orn = p.getQuaternionFromEuler(initial_obj_rpy)
    initial_obj_pos = [0.5, 0.0, 0]

    # Load object
    obj_dir = os.path.join(os.path.dirname(ShapeNetCore.__file__), args.obj_folder)
    with utils_sample.suppress_stdout():          # to suppress b3Warning           
        obj_id = pb.loadURDF(
            os.path.join(obj_dir, "model.urdf"),
            initial_obj_pos,
            initial_obj_orn,
            useFixedBase=True,
            flags=pb.URDF_INITIALIZE_SAT_FEATURES,
            globalScaling=args.scale
        )
        print(f'PyBullet object ID: {obj_id}')

    # Load object and get world frame coordinates.
    obj_path = os.path.join(obj_dir, "model.obj")
    mesh_original = utils_mesh._as_mesh(trimesh.load(obj_path))
    # Process object vertices to match thee transformations on the urdf file
    vertices_wrld = utils_mesh.rotate_pointcloud(mesh_original.vertices, initial_obj_rpy) * args.scale + initial_obj_pos
    mesh = trimesh.Trimesh(vertices=vertices_wrld, faces=mesh_original.faces)

    # Store processed mesh in deepsdf pose
    verts_deepsdf = utils_mesh.rotate_pointcloud(mesh_original.vertices, initial_obj_rpy)
    mesh_deepsdf = trimesh.Trimesh(vertices=verts_deepsdf, faces=mesh_original.faces)
    mesh_deepsdf.export(os.path.join(test_dir, 'mesh_deepsdf.obj'))

    # Instantiate pointcloud for DeepSDF prediction
    pointclouds_deepsdf = torch.tensor([]).view(0, 3).to(device)

    # Total signed distance
    sdf_gt = torch.tensor([]).view(0, 1).to(device)

    for num_sample in range(args.num_samples):

        # Get point clouds from the object
        pointcloud_deepsdf = get_synthetic_touch(mesh_deepsdf, radius=0.05)
        pointclouds_deepsdf = torch.vstack((pointclouds_deepsdf, pointcloud_deepsdf))

        # The sdf of points on the object surface is 0.
        sdf_gt = torch.vstack((sdf_gt, torch.zeros(size=(pointcloud_deepsdf.shape[0], 1)).to(device)))

        # Save pointclouds
        points_sdf = [pointclouds_deepsdf.detach().cpu(), sdf_gt.detach().cpu()]
        points_sdf_dir = os.path.join(test_dir, 'data', str(num_sample))
        if not os.path.isdir(points_sdf_dir):
            os.makedirs(points_sdf_dir)
        torch.save(points_sdf, os.path.join(points_sdf_dir, f'points_sdf.pt'))

    return test_dir


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
        "--scale", default=0.2, type=float, help="Scale of the object in simulation wrt the urdf object"
    )
    parser.add_argument(
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore' or 'PartNetMobility'"
    )
    parser.add_argument(
        "--obj_folder", type=str, default='', help="Object to reconstruct as obj_class/obj_category, e.g. 02818832/1aa55867200ea789465e08d496c0420f"
    )
    args = parser.parse_args()

    _ = main(args)