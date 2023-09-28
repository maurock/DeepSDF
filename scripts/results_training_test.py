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
import data.ShapeNetCoreV2urdf_test as ShapeNetCore_test
import data.ABC_test as ABC_test
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
from pytorch3d.loss import chamfer_distance
import random
from data_making.extract_touch_charts import load_environment
from utils.utils_metrics import earth_mover_distance, calculate_error_area, hausdorff_distance, calculate_fscore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Demo to reconstruct objects using tactile-gym.
"""
def main(args):
    # All object paths in the dataset
    if args.dataset == 'ShapeNetCore':
        dataset_folder = os.path.dirname(ShapeNetCore.__file__)
        suffix = '_ShapeNetCore_train'
    elif args.dataset == 'ShapeNetCore_test':
        dataset_folder = os.path.dirname(ShapeNetCore_test.__file__)
        suffix = '_ShapeNetCore_test'
    elif args.dataset == 'ABC_test':
        dataset_folder = os.path.dirname(ABC_test.__file__)
        suffix = '_ABC_test'
    else:
        raise ValueError('Dataset not recognised')
    full_paths = glob(os.path.join(dataset_folder, args.category, '*/'))
    obj_folders = [os.sep.join(full_path.split(os.sep)[-3:-1]) for full_path in full_paths]
    # For consistency:
    obj_folders = sorted(obj_folders)

    # Logging
    test_dir = os.path.join(os.path.dirname(runs_touch_sdf.__file__), datetime.now().strftime('%d_%m_%H%M%S_') + str(random.randint(0, 10000)) + suffix)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    log_path = os.path.join(test_dir, 'settings.txt')
    args_dict = vars(args)  # convert args to dict to write them as json
    with open(log_path, mode='a') as log:
        log.write('Settings:\n')
        log.write(json.dumps(args_dict).replace(', ', ',\n'))
        log.write('\n\n')

    metrics_path = os.path.join(test_dir, 'metrics.txt')  # log metrics in human readable format
    results_path = os.path.join(test_dir, 'results.npy')  # log metrics in numpy format

    # Load touch model
    touch_model = model_touch.Encoder().to(device)
    # Load weights for sdf model
    weights_path = os.path.join(os.path.dirname(runs_touch.__file__),  args.folder_touch, 'weights.pt')
    touch_model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    touch_model.eval()

    # Load sdf model
    sdf_model = model_sdf.SDFModelMulti(
        num_layers=8, 
        no_skip_connections=False,
        inner_dim=args.inner_dim,
        latent_size=args.latent_size,
        positional_encoding_embeddings=args.positional_encoding_embeddings).float().to(device)
    
    # Load weights for sdf model
    weights_path = os.path.join(os.path.dirname(runs_sdf.__file__), args.folder_sdf, 'weights.pt')
    sdf_model.load_state_dict(torch.load(weights_path, map_location=device))

    # Initial verts of the default touch chart
    chart_location = os.path.join(os.path.dirname(data.__file__), 'touch_chart.obj')
    initial_verts, initial_faces = utils_mesh.load_mesh_touch(chart_location)
    initial_verts = torch.unsqueeze(initial_verts, 0)

    # Instantiate grid coordinates for mesh extraction
    coords, grid_size_axis = utils_deepsdf.get_volume_coords(args.resolution)
    coords = coords.clone().to(device)
    coords_batches = torch.split(coords, 500000)

    # Get the average optimised latent code
    results_sdf_path = os.path.join(os.path.dirname(runs_sdf.__file__), args.folder_sdf, 'results.npy')
    results_sdf = np.load(results_sdf_path, allow_pickle=True).item()
    latent_code = results_sdf['train']['best_latent_codes']
    # Get average latent code (across dimensions)
    latent_code = torch.mean(torch.tensor(latent_code, dtype=torch.float32), dim=0).to(device)
    latent_code.requires_grad = True

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
    initial_obj_rpy = [np.pi/2 + args.change_orn[0], 0 + args.change_orn[1], -np.pi/2 + args.change_orn[2]]
    initial_obj_orn = p.getQuaternionFromEuler(initial_obj_rpy)
    initial_obj_pos = [0.5, 0.0, 0]

    results = dict()    
    for i in args.num_samples_extraction:
        results[i-1] = dict()
        results[i-1]['CD'] = []
        results[i-1]['EMD'] = []
        results[i-1]['area'] = []
        results[i-1]['F1'] = []
        results[i-1]['HD'] = []

    for idx, obj_folder in enumerate(obj_folders):

        # Reset simulation and reload the environment to avoid a silent bug where memory fills and visual rendering fails. 
        if idx > 0:
            pb.resetSimulation()
        
        pb, robot = load_environment(args, robot_config, pb)

        # Load object
        obj_dir = os.path.join(dataset_folder, obj_folder)

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

        verts_deepsdf = utils_mesh.rotate_pointcloud(mesh_original.vertices, initial_obj_rpy)
        mesh_deepsdf = trimesh.Trimesh(vertices=verts_deepsdf, faces=mesh_original.faces)

        # Ray: sqrt( (x1 - xc)**2 + (y1 - yc)**2)
        ray_hemisphere = utils_sample.get_ray_hemisphere(mesh)

        # Instantiate pointcloud for DeepSDF prediction
        pointclouds_deepsdf = torch.tensor([]).view(0, 3).to(device)

        # Total signed distance
        sdf_gt = torch.tensor([]).view(0, 1).to(device)

        # Deactivate collision between robot and object. Raycasting to extract point cloud still works.
        for link_idx in range(pb.getNumJoints(robot.robot_id)+1):
            pb.setCollisionFilterPair(robot.robot_id, obj_id, link_idx, -1, 0)

        ##############################################################################################
        # During the first stage, we collect tactile images, map them to point clouds, and store them.
        num_sample = 0

        while num_sample!=args.num_samples:

            robot.arm.worldframe_to_workframe([0.65, 0.0, 1.2], [0, 0, 0])[0]
            
            robot.results_at_touch_wrld = None

            hemisphere_random_pos, angles = utils_sample.sample_sphere(ray_hemisphere)

            # Move robot to random position on the hemisphere
            robot_sphere_wrld = mesh.bounding_box.centroid + np.array(hemisphere_random_pos)
            robot = utils_sample.robot_touch_spherical(robot, robot_sphere_wrld, initial_obj_pos, angles)

            # Check that the object is correctly sampled by checking that robot.stop_at_touch is not true 
            if robot.stop_at_touch:
                print("robot.stop_at_touch is true. Object not correctly sampled.")
                #pb.removeBody(robot.robot_id)
                continue
            
            # Check that the robot is touching the object and the avg colour pixel doesn't exceed a threshold
            camera = robot.get_tactile_observation()
            check_on_camera = utils_sample.check_on_camera(camera)
            if not check_on_camera:
                #pb.removeBody(robot.robot_id)
                continue

            # Check that the robot is touching the object and not other objects by ensuring that there is at least one valid point
            contact_pointcloud = utils_raycasting.filter_point_cloud(robot.results_at_touch_wrld, obj_id)
            check_on_contact_pointcloud = utils_sample.check_on_contact_pointcloud(contact_pointcloud, 1)
            if not check_on_contact_pointcloud:
                print(f'Point cloud shape is too small: {contact_pointcloud.shape[0]} points')
                #pb.removeBody(robot.robot_id)
                continue
            
            # Preprocess and store tactile image
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
            sdf_gt = torch.vstack((sdf_gt, torch.zeros(size=(pointcloud_deepsdf.shape[0], 1)).to(device)))

            # Add randomly sampled points from normals
            if args.augment_points_num > 0:
                # The sensor direction is given by the vectors pointing from the pointcloud to the TCP position
                # We need to first convert the TCP position to DeepSDF scale
                rpy_wrld = np.array(robot.arm.get_current_TCP_pos_vel_worldframe()[1]) # TCP orientation
                normal_wrk = np.array([[0, 0, 1]])
                normal_wrld = utils_mesh.rotate_pointcloud(normal_wrk, rpy_wrld)[0]
                TCP_pos_deepsdf = (robot.arm.get_current_TCP_pos_vel_worldframe()[0] -  0.01 * normal_wrld)  # move it slightly away from the object
                TCP_pos_deepsdf = (TCP_pos_deepsdf -  initial_obj_pos)/ args.scale # convert to DeepSDF scale
                sensor_dirs = TCP_pos_deepsdf - pointcloud_deepsdf_np
                
                # Estimate point cloud normals
                _, n = pcu.estimate_point_cloud_normals_knn(pointcloud_deepsdf_np, 32, view_directions=sensor_dirs)

                # Sample along normals and return points and distances
                pointcloud_along_norm_np, signed_distance_np = utils_sample.sample_along_normals(
                    std_dev=args.augment_points_std, pointcloud=pointcloud_deepsdf_np, normals=n, N=args.augment_points_num,
                    augment_multiplier_out=args.augment_multiplier_out)

                pointcloud_along_norm = torch.from_numpy(pointcloud_along_norm_np).float().to(device)
                sdf_normal_gt = torch.from_numpy(signed_distance_np).float().to(device)

                pointclouds_deepsdf = torch.vstack((pointclouds_deepsdf, pointcloud_along_norm))
                sdf_gt = torch.vstack((sdf_gt, sdf_normal_gt))

            if num_sample in [num-1 for num in args.num_samples_extraction]:
                # Infer latent code
                sdf_model.train()
                best_latent_code = sdf_model.infer_latent_code(args, pointclouds_deepsdf, sdf_gt,  None, latent_code)

                if args.finetuning:
                    best_weights = sdf_model.finetune(args, best_latent_code, pointclouds_deepsdf, sdf_gt, None)
                    sdf_model.load_state_dict(best_weights)
                
                sdf_model.eval()
                with torch.no_grad():
                    # Extract mesh obtained with the latent code optimised at inference
                    sdf = utils_deepsdf.predict_sdf(best_latent_code, coords_batches, sdf_model)
                
                try:
                    vertices_deepsdf, faces_deepsdf = utils_deepsdf.extract_mesh(grid_size_axis, sdf)
                except:
                    print('Mesh extraction failed')
                    num_sample += 1
                    continue

                # Compute Chamfer Distance
                # Get reconstructed meshes                
                reconstructed_mesh = trimesh.Trimesh(vertices_deepsdf, faces_deepsdf)

                if args.save_reconstruction:
                    # Create mesh folder if necessary
                    if not os.path.exists(os.path.join(test_dir, 'meshes')):
                        os.makedirs(os.path.join(test_dir, 'meshes'))
                    # Save reconstructed mesh
                    mesh_path = os.path.join(test_dir, 'meshes', f'mesh_{obj_folder.split(os.sep)[-1]}.obj')
                    trimesh.exchange.export.export_mesh(reconstructed_mesh, mesh_path, file_type='obj')                                    

                # Sample point cloud from both meshes
                original_pointcloud, _ = trimesh.sample.sample_surface(mesh_deepsdf, 4096)
                reconstructed_pointcloud, _ = trimesh.sample.sample_surface(reconstructed_mesh, 4096)
                
                # Get metrics
                cd = chamfer_distance(torch.tensor(np.array([original_pointcloud]), dtype=torch.float32),torch.tensor(np.array([reconstructed_pointcloud]), dtype=torch.float32))[0]
                
                emd = earth_mover_distance(original_pointcloud, reconstructed_pointcloud)

                error_area = calculate_error_area(reconstructed_mesh.vertices, reconstructed_mesh.faces, mesh_deepsdf.area)

                hd = hausdorff_distance(original_pointcloud, reconstructed_pointcloud)

                # For F1, we increase the number of points to 20000 as this is a measure of surface coverage
                original_pointcloud_f1, _ = trimesh.sample.sample_surface(mesh_deepsdf, 20000)
                reconstructed_pointcloud_f1, _ = trimesh.sample.sample_surface(reconstructed_mesh, 20000)
                f1, _, _ = calculate_fscore(original_pointcloud_f1, reconstructed_pointcloud_f1, args.threshold_f1)

                results[num_sample]['CD'].append(cd.item())
                results[num_sample]['EMD'].append(emd.item())
                results[num_sample]['area'].append(error_area)
                results[num_sample]['HD'].append(hd)
                results[num_sample]['F1'].append(f1)
                
                # Save results in a txt file
                with open(metrics_path, 'a') as log:
                    log.write(f'Obj id: {obj_folder}, Sample: {num_sample}, CD: {cd}, EMD: {emd}, Error area: {error_area}, HD: {hd}, F1: {f1}\n')

                # Save results
                np.save(results_path, results)

            # pb.removeBody(robot.robot_id)
            num_sample += 1


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
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore', 'ShapeNetCore_test', or 'PartNetMobility'"
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
        "--inner_dim", type=int, default=512, help="Inner dimensions of the network"
    )
    parser.add_argument(
        "--positional_encoding_embeddings", type=int, default=0, help="Number of embeddingsto use for positional encoding. If 0, no positional encoding is used."
    )
    parser.add_argument(
        "--category", default='*/', type=str, help="If default, loops through all the categories. Otherwise, specify the category, e.g. '02958343'"
    )
    parser.add_argument(
        "--resolution", type=int, default=15, help="Resolution of the extracted mesh"
    )
    parser.add_argument(
        "--num_samples_extraction", type=int, default=10, nargs='+', help="Number of samples on the objects. It can be a single number or a list of numbers, e.g. 10 20 30."
    )
    parser.add_argument(
        "--change_orn", type=float, default=[0, 0, 0], nargs='+', help="Change orientation of the object for ablation analysis."
    )
    parser.add_argument(
        "--save_reconstruction", default=False, action='store_true', help="Store reconstructed mesh"
    )
    parser.add_argument(
        "--epochs_finetuning", default=100, type=int, help="Number of epochs for latent code inference"
    )
    parser.add_argument(
        "--finetuning", default=False, action='store_true', help="Finetune the network after latent code inference."
    )
    parser.add_argument(
        "--lr_finetuning", type=float, default=0.0001, help="Learning rate for finetune"
    )
    parser.add_argument(
        "--threshold_f1", type=float, default=0.005, help="Threshold for computation of F-1 score"
    )
    args = parser.parse_args()

    # args.show_gui = True
    # args.folder_sdf ='10_08_181018'
    # args.lr_scheduler = True
    # args.epochs = 100
    # args.lr = 0.0005 
    # args.patience = 100 
    # args.resolution = 20 
    # args.num_samples = 5
    # args.num_samples_extraction = [5]
    # args.mode_reconstruct = 'fixed'
    # args.langevin_noise = 0.0
    # args.dataset = 'ABC_test'
    # args.folder_touch = '12_08_1122'
    # args.category = 'data'
    # args.finetuning = False
    # args.epochs_finetuning = 10
    # args.clamp = True
    # args.clamp_valur = 0.5

    main(args)