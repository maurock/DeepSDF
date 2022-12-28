import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import pkgutil
import os
from cri_robot_arm import CRIRobotArm
from tactile_gym.assets import add_assets_path
from utils import utils_sample, utils_mesh, utils_raycasting
import argparse
from glob import glob
import data.objects as objects

def main(args):
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

    # load the environment
    plane_id = pb.loadURDF(
        add_assets_path("shared_assets/environment_objects/plane/plane.urdf")
    )

    # robot configuration
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

    # Get list of object indices
    list_objects = [filepath.split('/')[-2] for filepath in glob(os.path.join(os.path.dirname(objects.__file__), '*/'))]  

    # Initialise dict with arrays to store.
    data = {
        "verts": np.array([]).reshape(0, 75), # verts of touch charts (25) flattened
        "tactile_imgs": np.array([], dtype=np.float32).reshape(0, 1, 256, 256),
        "pointclouds": np.array([], dtype=np.float32).reshape(0, 500, 3),   # fixed dimension touch chart pointcloud (workframe)
        "rot_M_wrld_list": np.array([], dtype=np.float32).reshape(0, 3, 3),      # rotation matrix (work wrt worldframe)
        "pos_wrld_list": np.array([]).reshape(0, 3) , # TCP pos (worldframe)
        "pos_wrk_list": np.array([], dtype=np.float32).reshape(0, 3),   # TCP pos (worldframe)
        "obj_index": np.array([], dtype=np.float32).reshape(0, 1),
        "initial_pos": np.array([], dtype=np.float32).reshape(0, 3)
    }

    for idx, obj_index in enumerate(list_objects[1:10]): 
        print(f"Collecting data... Object index: {obj_index} \t {idx+1}/{len(list_objects)} ")

        # Load object
        initial_obj_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])

        with utils_sample.suppress_stdout():          # to suppress b3Warning   
            # Calculate initial z position for object
            obj_initial_z = utils_mesh.calculate_initial_z(obj_index, args.scale)
        
            initial_obj_pos = [0.65, 0.0, obj_initial_z]
            
            obj_id = pb.loadURDF(
                os.path.join(os.path.dirname(objects.__file__), f"{obj_index}/mobility.urdf"),
                initial_obj_pos,
                initial_obj_orn,
                useFixedBase=True,
                flags=pb.URDF_INITIALIZE_SAT_FEATURES,
                globalScaling=args.scale
            )
            print(f'PyBullet object ID: {obj_id}')

        #robot.arm.worldframe_to_workframe([0.65, 0.0, 1.2], [0, 0, 0])[0]

        # Get world frame coordinates. These do not take into account the initial obj position.
        _, vertices_wrld = pb.getMeshData(obj_id, 0)   
        initial_rpy = [np.pi / 2, 0, 0]   # WHY DO I NEED THIS ARBITRARY ORN INSTEAD OF OBJ ORN?
        vertices_wrld = utils_mesh.rotate_pointcloud(np.array(vertices_wrld), initial_rpy) + initial_obj_pos

        # Ray: sqrt( (x1 - xc)**2 + (y1 - yc)**2)
        ray_hemisphere = utils_sample.get_ray_hemisphere(vertices_wrld, initial_obj_pos) 

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
            
            # Filter points with information about contact, make sure there are at least 500 valid ones
            contact_pointcloud = utils_raycasting.filter_point_cloud(robot.results_at_touch_wrld)
            if contact_pointcloud.shape[0] < 500:
                print(f'Point cloud shape is too small: {contact_pointcloud.shape[0]} points')
                pb.removeBody(robot.robot_id)
                continue

            # Sample 500 random points among the contact ones 
            random_indices = np.random.choice(contact_pointcloud.shape[0], 500)
            sampled_pointcloud_wrld = contact_pointcloud[random_indices]
            # Centre points on origin and convert to workframe
            tcp_pos_wrld, tcp_rpy_wrld, _, _, _ = robot.arm.get_current_TCP_pos_vel_worldframe()
            sampled_pointcloud_wrld = sampled_pointcloud_wrld - tcp_pos_wrld
            sampled_pointcloud_wrk = utils_mesh.rotate_pointcloud_inverse(sampled_pointcloud_wrld, tcp_rpy_wrld)
            sampled_pointcloud_wrk = sampled_pointcloud_wrk[None, :, :]  # increase dim for stacking
            data['pointclouds'] = np.vstack((data['pointclouds'], sampled_pointcloud_wrk))

            # Full pointcloud to 25 vertices. By default, vertices are converted to workframe.
            verts_wrk = utils_raycasting.pointcloud_to_vertices_wrk(contact_pointcloud, robot, args)
            # if verts_wrk.shape[0] != 25:
            #     print('Point cloud to vertices could not extract 25 vertices.')
            #     pb.removeBody(robot.robot_id)
            #     continue
            verts_ravel_wrk = np.asarray(verts_wrk, dtype=np.float32).ravel()
            data['verts'] = np.vstack((data['verts'], verts_ravel_wrk))

            # Store world position of the TCP
            data['pos_wrld_list'] = np.vstack((data['pos_wrld_list'], robot.coords_at_touch_wrld))

            # Store tactile images
            camera = robot.get_tactile_observation()[np.newaxis, :, :]
            # Conv2D requires [batch, channels, size1, size2] as input
            tactile_imgs_norm = np.expand_dims(camera, 0) / 255     # normalize tactile images
            data['tactile_imgs'] = np.vstack((data['tactile_imgs'], tactile_imgs_norm))

            # Store TCP position in work frame
            pos_wrk = robot.arm.get_current_TCP_pos_vel_workframe()[0]
            data['pos_wrk_list'] = np.vstack((data['pos_wrk_list'], pos_wrk))

            # Store TCP orientation in world frame
            rot_Q_wrld = robot.arm.get_current_TCP_pos_vel_worldframe()[2]
            rot_M_wrld = np.array(pb.getMatrixFromQuaternion(rot_Q_wrld)).reshape(1, 3, 3)
            data['rot_M_wrld_list'] = np.vstack((data['rot_M_wrld_list'], rot_M_wrld))

            data['obj_index'] = np.vstack((data['obj_index'], obj_index))

            data['initial_pos'] = np.vstack((data['initial_pos'], initial_obj_pos))

            utils_sample.save_touch_charts(data)

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
        "--scale", default=0.1, type=float, help="Scale of the object in simulation wrt the urdf object"
    )
    args = parser.parse_args()

    main(args)