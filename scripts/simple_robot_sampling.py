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
import results 
import matplotlib.pyplot as plt
import trimesh

def main(args):
    time_step = 1. / 1  # low for small objects

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
    #list_objects = ['/Users/ri21540/Documents/PhD/Code/DeepSDF/data/ShapeNetCoreV2/02942699/1ab3abb5c090d9b68e940c4e64a94e1e/']
    list_objects = ['/Users/ri21540/Documents/PhD/Code/TEMP/STL']

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

    for idx, obj_index in enumerate(list_objects): 
        print(f"Collecting data... Object index: {obj_index} \t {idx+1}/{len(list_objects)} ")

        # Load object
        initial_obj_orn = p.getQuaternionFromEuler([np.pi/2, 0, -np.pi/2])

        with utils_sample.suppress_stdout():          # to suppress b3Warning   
            # Calculate initial z position for object
            #obj_initial_z = utils_mesh.calculate_initial_z(obj_index, args.scale, args.dataset)
        
            initial_obj_rpy = [np.pi/2, 0, -np.pi/2]
            initial_obj_orn = p.getQuaternionFromEuler(initial_obj_rpy)
            initial_obj_pos = [0.5, 0.0, 0]
            
            obj_id = pb.loadURDF(
                #os.path.join(os.path.dirname(objects.__file__), f"{obj_index}/model.urdf"),
                os.path.join('/Users/ri21540/Documents/PhD/Code/DeepSDF/data/ShapeNetCoreV2urdf/bowl/c0f57c7f98d2581c744a455c7eef0ae5/model.urdf'),
                initial_obj_pos,
                initial_obj_orn,
                useFixedBase=True,
                flags=pb.URDF_INITIALIZE_SAT_FEATURES,
                globalScaling=args.scale
            )
            print(f'PyBullet object ID: {obj_id}')

        #robot.arm.worldframe_to_workframe([0.65, 0.0, 1.2], [0, 0, 0])[0]

        # Store the scaled, rotated, and translated mesh vertices
        # np.save(os.path.join(os.path.dirname(results.__file__), f'checkpoints/vertices_wrld_{obj_index}.npy'), vertices_wrld)

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

        # robot.arm.worldframe_to_workframe([0.65, 0.0, 1.2], [0, 0, 0])[0]
        
        while True:
            pb.stepSimulation()
        
        # if args.show_gui:
        #     time.sleep(1)


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
        "--scale", default=0.4, type=float, help="Scale of the object in simulation wrt the urdf object"
    )
    parser.add_argument(
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore' or 'PartNetMobility'"
    )
    args = parser.parse_args()

    args.show_gui = True
    args.scale = 0.2

    main(args)