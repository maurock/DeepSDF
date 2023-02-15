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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Demo to reconstruct objects using tactile-gym.
"""
@profile
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

    # Load sdf model
    sdf_model = model_sdf.SDFModelMulti(num_layers=8, no_skip_connections=False).float().to(device)
    
    # Load weights for sdf model
    weights_path = os.path.join(os.path.dirname(runs_sdf.__file__), args.folder_sdf, 'weights.pt')
    sdf_model.load_state_dict(torch.load(weights_path, map_location=device))
    sdf_model.eval()

    # Load touch model
    touch_model = model_touch.Encoder().to(device)
    # Load weights for sdf model
    weights_path = os.path.join(os.path.dirname(runs_touch.__file__),  args.folder_touch, 'weights.pt')
    touch_model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    touch_model.eval()

    # Initial verts of the default touch chart
    chart_location = os.path.join(os.path.dirname(data.__file__), 'touch_chart.obj')
    initial_verts, initial_faces = utils_mesh.load_mesh_touch(chart_location)
    initial_verts = torch.unsqueeze(initial_verts, 0)

    time_step = 1. / 960  # low for small objects

    if args.show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  

        # set debug camera position
        cam_dist = 0.5
        cam_yaw = 90
        cam_pitch = -25
        cam_pos = [0.65, 0, 0.025]
        pb.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_pos)

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

    # Robot configuration
    robot_config = {
        # workframe
        'workframe_pos': [0.65, 0.0, 0.35], # relative to world frame
        'workframe_rpy': [-np.pi, 0.0, np.pi / 2], # relative to world frame
        'image_size': [256, 256],
        'arm_type': 'ur5',
        # sensor
        't_s_type': 'standard',
        't_s_core': 'no_core',
        't_s_name': 'tactip',
        't_s_dynamics': {},
        'show_gui': args.show_gui,
        'show_tactile': args.show_tactile,
        'nx': 50,
        'ny': 50
    }
    
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

    # Ray: sqrt( (x1 - xc)**2 + (y1 - yc)**2)
    ray_hemisphere = utils_sample.get_ray_hemisphere(mesh)

    # Instantiate pointcloud for DeepSDF prediction
    pointclouds_deepsdf = torch.tensor([]).view(0, 3).to(device)

    # Instantiate grid coordinates for mesh extraction
    coords, grid_size_axis = utils_deepsdf.get_volume_coords(args.resolution)
    coords = coords.clone().to(device)
    coords_batches = torch.split(coords, 100000)

    # Instantiate variables to store checkpoints
    checkpoint_dict = dict()

    for num_sample in range(args.num_samples):

        robot = CRIRobotArm(
            pb,
            workframe_pos = robot_config['workframe_pos'],
            workframe_rpy = robot_config['workframe_rpy'],
            image_size = [256, 256],
            arm_type = "ur5",
            t_s_type = robot_config['t_s_type'],
            t_s_core = robot_config['t_s_core'],
            t_s_name = robot_config['t_s_name'],
            t_s_dynamics = robot_config['t_s_dynamics'],
            show_gui = args.show_gui,
            show_tactile = args.show_tactile
        )
        
        # Set pointcloud grid
        robot.nx = robot_config['nx']
        robot.ny = robot_config['ny']

        robot.arm.worldframe_to_workframe([0.65, 0.0, 1.2], [0, 0, 0])[0]
        
        robot.results_at_touch_wrld = None

        # Sample random position on the hemisphere
        hemisphere_random_pos, angles = utils_sample.sample_sphere(ray_hemisphere)

        # Move robot to random position on the hemisphere
        robot_sphere_wrld = np.array(initial_obj_pos) + np.array(hemisphere_random_pos)
        robot = utils_sample.robot_touch_spherical(robot, robot_sphere_wrld, initial_obj_pos, angles)
        
        # Check that the robot is touching the object
        camera = robot.get_tactile_observation()
        if np.mean(camera) == 0.0:
            pb.removeBody(robot.robot_id)
            continue
        # Conv2D requires [batch, channels, size1, size2] as input
        tactile_imgs_norm = camera[np.newaxis, np.newaxis, :, :] / 255 
        tactile_img = torch.tensor(tactile_imgs_norm, dtype=torch.float32).to(device)

        # Predict vertices from tactile image (TCP frame, Sim scale)
        predicted_verts = touch_model(tactile_img, initial_verts)[0]

        # Create mesh from predicted vertices
        predicted_mesh = trimesh.Trimesh(predicted_verts.detach().cpu(), initial_faces.cpu())

        # Sample pointcloud from predicted mesh in Sim coordinates (TCP frame, Sim scale)
        predicted_pointcloud = utils_mesh.mesh_to_pointcloud(predicted_mesh, 500)

        # Translate to world frame
        pos_wrld = robot.coords_at_touch_wrld[None, :]
        rot_Q_wrld = robot.arm.get_current_TCP_pos_vel_worldframe()[2]
        rot_M_wrld = np.array(pb.getMatrixFromQuaternion(rot_Q_wrld)).reshape(1, 3, 3)
        predicted_pointcloud_wrld = utils_mesh.translate_rotate_mesh(pos_wrld, rot_M_wrld, predicted_pointcloud[None, :, :], initial_obj_pos)

        # Rescale from Sim scale to DeepSDF scale
        pointcloud_deepsdf_np = (predicted_pointcloud_wrld / args.scale)[0]  # shape (n, 3)

        # Concatenate predicted pointclouds of the touch charts from all samples
        pointcloud_deepsdf = torch.from_numpy(pointcloud_deepsdf_np).float().to(device)  # shape (n, 3)
        pointclouds_deepsdf = torch.vstack((pointclouds_deepsdf, pointcloud_deepsdf))   
        
        # The sdf of points on the object surface is 0.
        sdf_gt = torch.zeros(size=(pointclouds_deepsdf.shape[0], 1)).to(device)

        # Add randomly sampled points from normals
        if args.augment_points:
            # Everything in this block is in deepsdf scale
            # Vectors pointing from the points to the TCP position
            sensor_dirs = robot.arm.get_current_TCP_pos_vel_worldframe()[0] - pointcloud_deepsdf_np
            
            # Estimate point cloud normals
            _, n = pcu.estimate_point_cloud_normals_knn(pointcloud_deepsdf_np, 32, view_directions=sensor_dirs)

            # Sample along normals and return points and distances
            pointcloud_along_norm_np, signed_distance_np = utils_sample.sample_along_normals(
                std_dev=args.augment_points_std, pointcloud=pointcloud_deepsdf_np, normals=n, N=args.augment_points_num)
            pointcloud_along_norm = torch.from_numpy(pointcloud_along_norm_np).float().to(device)
            signed_distance = torch.from_numpy(signed_distance_np).float().to(device)

            pointclouds_deepsdf = torch.vstack((pointclouds_deepsdf, pointcloud_along_norm))
            sdf_gt = torch.vstack((sdf_gt, signed_distance))

        # Infer latent code
        log_tensorboard_path = os.path.join(test_dir, str(num_sample))
        if not os.path.isdir(log_tensorboard_path):
            os.makedirs(log_tensorboard_path)
        writer = SummaryWriter(log_dir=os.path.join(test_dir, str(num_sample)))
        best_latent_code = sdf_model.infer_latent_code(args, pointclouds_deepsdf, sdf_gt, writer)

        # Predict sdf values from pointcloud
        # input_deepsdf = torch.cat((pointclouds_deepsdf, best_latent_code.repeat(pointclouds_deepsdf.shape[0], 1)), dim=1)
        # predicted_coords = sdf_model(input_deepsdf)

        # Extract mesh obtained with the latent code optimised at inference
        sdf = utils_deepsdf.predict_sdf(best_latent_code, coords_batches, sdf_model)
        vertices_deepsdf, faces_deepsdf = utils_deepsdf.extract_mesh(grid_size_axis, sdf)

        # Save mesh
        checkpoint_dict[num_sample] = dict()
        checkpoint_dict[num_sample]['mesh'] = [vertices_deepsdf, faces_deepsdf]
        checkpoint_dict[num_sample]['pointcloud'] = [utils_mesh.rotate_pointcloud(mesh_original.vertices, initial_obj_rpy), pointclouds_deepsdf.cpu()]

        pb.removeBody(robot.robot_id)

    pb.removeBody(obj_id)

    # Save checkpoint
    checkpoint_path = os.path.join(test_dir, 'checkpoint_dict.npy')
    np.save(checkpoint_path, checkpoint_dict)

    if args.show_gui:
        time.sleep(1)


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
        "--folder_sdf", default=0, type=str, help="Folder containing the sdf model weights"
    )
    parser.add_argument(
        "--folder_touch", default=0, type=str, help="Folder containing the touch model weights"
    )
    parser.add_argument(
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore' or 'PartNetMobility'"
    )

    # Argument for inference
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
        "--obj_folder", type=str, default='', help="Object to reconstruct as obj_class/obj_category, e.g. 02818832/1aa55867200ea789465e08d496c0420f"
    )
    parser.add_argument(
        "--augment_points", default=False, action='store_true', help="Estimate point cloud normals and sample points along them (negative and positive direction)"
    )
    parser.add_argument(
        "--augment_points_std", default=0.01, type=float, help="Standard deviation of the Gaussian used to sample points along normals (if augment_points is True)"
    )
    parser.add_argument(
        "--augment_points_num", default=5, type=int, help="Number of points to sample along normals (if augment_points is True)"
    )
    args = parser.parse_args()

    # args.show_gui =True
    # args.num_samples =20
    # args.folder_sdf ='23_01_095414'
    # args.folder_touch ='14_02_1521' 
    # args.obj_folder ='lamp/c3277019e57251cfb784faac204319d9' 
    # args.lr_scheduler = True
    # args.epochs =1000 
    # args.lr =0.00005 
    # args.patience =100 
    # args.resolution =256 
    # args.augment_points=True

    main(args)