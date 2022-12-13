import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import pkgutil
import os
from cri_robot_arm import CRIRobotArm
from tactile_gym.assets import add_assets_path
from utils import utils_sample
from utils import utils_mesh
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

    if args.show_gui:
        # set debug camera position
        cam_dist = 0.5
        cam_yaw = 90
        cam_pitch = -25
        cam_pos = [0.65, 0, 0.025]
        pb.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_pos)

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
    
    list_objects = [filepath.split('/')[-1] for filepath in glob(os.path.join(os.path.dirname(objects.__file__), '*'))]
    list_objects.remove('__init__.py')
    list_objects.remove('__pycache__')

    # initialise dict with arrays to store. They'll be processed in samples_utils.save_touch_charts
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
        print(f"Collecting data... Object index: {obj_index}     {idx+1}/{len(list_objects)} ")

        # Load robot
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

        # Load object
        stimulus_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])

        with utils_sample.suppress_stdout():          # to suppress b3Warning
            
            obj_initial_z = utils_mesh.get_mesh_z(obj_index, args.scale)
        
            stimulus_pos = [0.65, 0.0, obj_initial_z]
            
            obj_id = pb.loadURDF(
                #os.path.join('/Users/ri21540/Documents/PhD/Code/tactile_gym_sim2real_dev_Mauro/tactile_gym_sim2real/data_collection/sim/stimuli/edge_stimuli/square/square.urdf'),
                os.path.join(os.path.dirname(objects.__file__), f"{obj_index}/mobility.urdf"),
                stimulus_pos,
                stimulus_orn,
                useFixedBase=True,
                flags=pb.URDF_INITIALIZE_SAT_FEATURES,
                globalScaling=args.scale
            )
            print(f'PyBullet object ID: {obj_id}')

        robot.arm.worldframe_to_workframe([0.65, 0.0, 1.2], [0, 0, 0])[0]

        #mesh_list, tactile_imgs, pointcloud_list, obj_index, rot_M_wrld_list, pos_wrld_list, pos_wrk_list  =
        data = utils_sample.spherical_sampling(
            robot=robot,
            obj_id=obj_id, 
            initial_pos=stimulus_pos,
            initial_orn=stimulus_orn,
            args=args,
            obj_index=obj_index,
            robot_config=robot_config,
            data=data
        )

        #utils_mesh.save_touch_charts(mesh_list, tactile_imgs, pointcloud_list, rot_M_wrld_list, pos_wrld_list, pos_wrk_list, stimulus_pos, path)
        pb.removeBody(obj_id)
        pb.removeBody(robot.robot_id)
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
        "--debug_show_full_mesh", default=False, action='store_true', help="Show mesh obtained from first raycasting"
    )
    parser.add_argument(
        "--debug_show_mesh_wrk", default=False, action='store_true', help="Show mesh obtained from applying the pivot ball technique on 25 vertices wrt workframe"
    )
    parser.add_argument(
        "--debug_show_mesh_wrld", default=False, action='store_true', help="Show mesh obtained from applying the pivot ball technique on 25 vertices wrt worldframe"
    )
    parser.add_argument(
        "--debug_contact_points", default=False, action='store_true', help="Show contact points on Plotly"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samplings on the objects"
    )
    parser.add_argument(
        "--debug_rotation", default=False, action='store_true', help="Store data to see if rotation works"
    )
    parser.add_argument(
        "--render_scene", default=False, action='store_true', help="Render scene at touch"
    )
    parser.add_argument(
        "--scale", default=0.1, type=float, help="Scale of the object in simulation wrt the urdf object"
    )
    args = parser.parse_args()

    main(args)