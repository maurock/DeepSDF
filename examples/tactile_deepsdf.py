import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import pkgutil
import os
from cri_robot_arm import CRIRobotArm
from tactile_gym.assets import add_assets_path
from utils import utils_sample, utils_mesh, utils_deepsdf
import argparse
from glob import glob
import data.objects as objects
from model import model_sdf, model_touch
import torch
import data
from results import runs_touch, runs_sdf
import trimesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Demo to reconstruct objects using tactile-gym.
"""

def main(args):
    # Load sdf model
    sdf_model = model_sdf.SDFModelMulti(num_layers=8, no_skip_connections=False).to(device)
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
    plane_id = pb.loadURDF(
        add_assets_path("shared_assets/environment_objects/plane/plane.urdf")
    )

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

    # Load object
    with utils_sample.suppress_stdout():          # to suppress b3Warning   
        initial_obj_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])

        # Calculate initial z position for object
        obj_initial_z = utils_mesh.calculate_initial_z(args.obj_index, args.scale_sim)
    
        initial_obj_pos = [0.65, 0.0, obj_initial_z]
        
        obj_id = pb.loadURDF(
            os.path.join(os.path.dirname(objects.__file__), f"{args.obj_index}/mobility.urdf"),
            initial_obj_pos,
            initial_obj_orn,
            useFixedBase=True,
            flags=pb.URDF_INITIALIZE_SAT_FEATURES,
            globalScaling=args.scale_sim
        )
        print(f'PyBullet object ID: {obj_id}')

    # Get world frame coordinates. These do not take into account the initial obj position.
    _, vertices_wrld = pb.getMeshData(obj_id, 0)   
    initial_rpy = [np.pi / 2, 0, 0]   # WHY DO I NEED THIS ARBITRARY ORN INSTEAD OF OBJ ORN?
    vertices_wrld = utils_mesh.rotate_pointcloud(np.array(vertices_wrld), initial_rpy) + initial_obj_pos

    # Ray: sqrt( (x1 - xc)**2 + (y1 - yc)**2)
    ray_hemisphere = utils_sample.get_ray_hemisphere(vertices_wrld, initial_obj_pos) 

    # Instantiate pointcloud for DeepSDF prediction
    pointlcoud_deepsdf = torch.tensor([]).view(0, 3).to(device)

    for _ in range(args.num_samples):

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
        hemisphere_random_pos, angles = utils_sample.sample_hemisphere(ray_hemisphere)

        # Move robot to random position on the hemisphere
        robot_sphere_wrld = np.array(initial_obj_pos) + np.array(hemisphere_random_pos)
        robot = utils_sample.robot_touch_spherical(robot, robot_sphere_wrld, initial_obj_pos, angles)

        # If not contact points, continue. 
        if robot.results_at_touch_wrld is None:
            pb.removeBody(robot.robot_id)
            continue
        
        # Store tactile images
        camera = robot.get_tactile_observation()[np.newaxis, :, :]
        # Conv2D requires [batch, channels, size1, size2] as input
        tactile_img = np.expand_dims(camera, 0) / 255     # normalise tactile images
        tactile_img = torch.from_numpy(tactile_img).float().to(device)

        # Predict vertices from tactile image
        predicted_verts = touch_model(tactile_img, initial_verts)[0]

        # Create mesh from predicted vertices
        predicted_mesh = trimesh.Trimesh(predicted_verts.detach().numpy(), initial_faces)

        # Sample pointcloud from predicted mesh in Sim coordinates
        predicted_pointcloud = utils_mesh.mesh_to_pointcloud(predicted_mesh, 500)

        # Rescale from Sim to DeepSDF coordinates
        pointcloud_scaled = utils_deepsdf.scale_sim_to_deepsdf(predicted_pointcloud, args.scale_sim, args.scale_deepsdf)
        pointcloud_scaled = torch.from_numpy(pointcloud_scaled).float().to(device)

        # Concatenate pointclouds
        pointcloud_deepsdf = torch.vstack((pointlcoud_deepsdf, pointcloud_scaled))   
        
        # Predict sdf values from pointcloud
        sdf_model(pointcloud_deepsdf)

        # Infer latent code
        


        # 
        
        # # Filter points with information about contact, make sure there are at least 500 valid ones
        # contact_pointcloud = utils_raycasting.filter_point_cloud(robot.results_at_touch_wrld)
        # if contact_pointcloud.shape[0] < 500:
        #     print('Point cloud shape is too small')
        #     pb.removeBody(robot.robot_id)
        #     continue
        
        # # Sample 500 random points among the contact ones 
        # random_indices = np.random.choice(contact_pointcloud.shape[0], 500)
        # sampled_pointcloud_wrld = contact_pointcloud[random_indices]
        # # Centre points on origin and convert to workframe
        # tcp_pos_wrld, tcp_rpy_wrld, _, _, _ = robot.arm.get_current_TCP_pos_vel_worldframe()
        # sampled_pointcloud_wrld = sampled_pointcloud_wrld - tcp_pos_wrld
        # sampled_pointcloud_wrk = utils_mesh.rotate_pointcloud_inverse(sampled_pointcloud_wrld, tcp_rpy_wrld)
        # sampled_pointcloud_wrk = sampled_pointcloud_wrk[None, :, :]  # increase dim for stacking
        # data['pointclouds'] = np.vstack((data['pointclouds'], sampled_pointcloud_wrk))

        # # Full pointcloud to 25 vertices. By default, vertices are converted to workframe.
        # verts_wrk = utils_raycasting.pointcloud_to_vertices_wrk(contact_pointcloud, robot, args)
        # verts_ravel_wrk = np.asarray(verts_wrk, dtype=np.float32).ravel()
        # data['verts'] = np.vstack((data['verts'], verts_ravel_wrk))

        # # Store world position of the TCP
        # data['pos_wrld_list'] = np.vstack((data['pos_wrld_list'], robot.coords_at_touch_wrld))


        # # Store TCP position in work frame
        # pos_wrk = robot.arm.get_current_TCP_pos_vel_workframe()[0]
        # data['pos_wrk_list'] = np.vstack((data['pos_wrk_list'], pos_wrk))

        # # Store TCP orientation in world frame
        # rot_Q_wrld = robot.arm.get_current_TCP_pos_vel_worldframe()[2]
        # rot_M_wrld = np.array(pb.getMatrixFromQuaternion(rot_Q_wrld)).reshape(1, 3, 3)
        # data['rot_M_wrld_list'] = np.vstack((data['rot_M_wrld_list'], rot_M_wrld))

        # data['obj_index'] = np.vstack((data['obj_index'], obj_index))

        # data['initial_pos'] = np.vstack((data['initial_pos'], initial_obj_pos))

        pb.removeBody(robot.robot_id)

    pb.removeBody(obj_id)

    if args.show_gui:
        time.sleep(1)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
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
        "--scale_sim", default=0.1, type=float, help="Scale of the object in simulation wrt the urdf object"
    )
    parser.add_argument(
        "--scale_deepsdf", default=1.414, type=float, help="Scale of the DeepSDF mesh wrt the urdf object"
    )
    parser.add_argument(
        "--obj_index", default=0, type=str, help="Index of the object to reconstruct"
    )
    parser.add_argument(
        "--folder_sdf", default=0, type=str, help="Folder containing the sdf model weights"
    )
    parser.add_argument(
        "--folder_touch", default=0, type=str, help="Folder containing the touch model weights"
    )
    args = parser.parse_args()

    args.show_gui = True
    args.folder_sdf = '25_11_084304'
    args.folder_touch = '14_12_1508'
    args.obj_index = '3398'

    main(args)