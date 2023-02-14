import time
import numpy as np
import pybullet as pb
from utils import utils_raycasting, utils_mesh
import os
from scipy.spatial.transform import Rotation as R
import sys
from cri_robot_arm import CRIRobotArm
from utils import utils_mesh
import results
from contextlib import contextmanager
import trimesh

def sample_sphere(r):
    """
    Uniform sampling on a hemisphere.
    Parameter:
        - r: radius
    Returns:
        - [x, y, z]: list of points in world frame
        - [phi, theta]: phi is horizontal (0, pi/2), theta is vertical (0, pi/2) 
    """
    phi = 2 * np.pi * np.random.uniform()
    theta = np.arccos(1 - 2 * np.random.uniform())
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    coords = [x, y, z]
    angles = [phi, theta]

    return coords, angles


def _debug_plot_sphere(r, origin):
    """ 
    Draw a sphere (made of points) around the object
    """
    coords_array = np.array([]).reshape(0, 3)
    for i in range(0, 1000):
        coords, _ = sample_hemisphere(r)
        coords = coords + np.array(origin)
        coords_array = np.vstack((coords_array, coords))
    color_array = np.full(shape=coords_array.shape, fill_value=np.array([0, 72, 255])/255)
    pb.addUserDebugPoints(
            pointPositions=coords_array, 
            pointColorsRGB=color_array,
            pointSize=2)


def sphere_orn_wrld(robot, origin, angles):
    """
    Params:
        - angles: list [phi, theta], where phi is horizontal, theta is vertical
    The rotation matrix was taken from https://en.wikipedia.org/wiki/Spherical_coordinate_system
    and modified by swapping the x and z axes, and negating the z axis so that the robot goes towards the object.
    """
    phi, theta = angles
    rot_M = np.array([  [-np.sin(phi), np.cos(theta) * np.cos(phi), -np.sin(theta)*np.cos(phi)],
                        [np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta) * np.sin(phi)],
                        [0, -np.sin(theta), -np.cos(theta)]])
    # Draw sensor's frame for debug purposes
    # pb.addUserDebugLine(origin, (origin + rot_M[:, 0]), lineColorRGB=[1,0,0], lifeTime=20)
    # pb.addUserDebugLine(origin, (origin + rot_M[:, 1]), lineColorRGB=[0,1,0], lifeTime=20)
    # pb.addUserDebugLine(origin, (origin + rot_M[:, 2]), lineColorRGB=[0,0,1], lifeTime=20)
    r = R.from_matrix(rot_M)
    orn = r.as_euler('xyz')
    return orn


def move_wrld_to_work(robot, pos_wrld, orn_wrld=[3.14, 0, 1.57]):
    pos_wrk, orn_wrk = robot.arm.worldframe_to_workframe(pos_wrld, orn_wrld)
    robot.move_linear(pos_wrk, orn_wrk)

def robot_touch_spherical(robot, robot_sphere_wrld, initial_pos, angles, max_height_wrld=0.2):
    """Given x-y coordinates in the worldframe, the robot first moves to a high position and then sample the object's surface"""
    
    # go to rest position
    high_wrld = np.array(initial_pos) + np.array([0, 0, max_height_wrld])
    move_wrld_to_work(robot, high_wrld)
    
    # Go to position on sphere
    orn_wrld = sphere_orn_wrld(robot, robot_sphere_wrld, angles)
    move_wrld_to_work(robot, robot_sphere_wrld, orn_wrld)

    robot.stop_at_touch = True

    # Sample the object
    move_wrld_to_work(robot, initial_pos, orn_wrld)    # This does not work in DIRECT mode for a problem in the backgroun sensor image
    robot.results_at_touch_wrld = utils_raycasting.get_contact_points(robot.arm.get_current_TCP_pos_vel_worldframe(), robot._pb, robot.t_s_name, robot.nx, robot.ny)
    
    robot.stop_at_touch = False

    return robot


def spherical_sampling(robot, obj_id, initial_pos, initial_orn, args, obj_index, robot_config, data, num_points=2000, mesh=None, obj=False):
    """
    This method computes the max and min coordinates and samples randomly within these boundaries.
    If the mesh is loaded as .urdf, we use pybullet.getMesh(). If it is a .obj, we pass a Trimesh object and extracts the vertices.

    Return:
        - samples_list = np.array containing list of full pointclouds at touch (variable number of points)
        - mesh_list = list containing open3d.geometry.TriangleMesh (25 vertices and faces of the local geometry at touch site)
    """
    if obj:
        vertices_wrld = mesh.vertices
        vertices_wrld = np.array(vertices_wrld) * args.scale + initial_pos   # add initial obj position and scale
    else:
        # worldframe coordinates. These do not take into account the initial obj position.
        num_vertices, vertices_wrld = pb.getMeshData(obj_id, 0)   
        initial_rpy = [np.pi / 2, 0, 0]   # WHY DO I NEED THIS ARBITRARY ORN INSTEAD OF OBJ ORN?
        vertices_wrld = utils_mesh.rotate_pointcloud(np.array(vertices_wrld), initial_rpy) + initial_pos

    # Get min and max world object coordinates. 
    max_coords = [ np.amax(vertices_wrld[:,0]), np.amax(vertices_wrld[:,1]), np.amax(vertices_wrld[:,2]) ]

    # Ray: sqrt( (x1 - xc)**2 + (y1 - yc)**2)
    ray_hemisphere = 1.5 * np.sqrt((max_coords[0] - initial_pos[0])**2 + (max_coords[1] - initial_pos[1])**2 + (max_coords[2] - initial_pos[2])**2)

    for _ in range(args.num_samples):

        pb.removeBody(robot.robot_id)
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
        
        robot.results_at_touch_wrld = None

        # Sample random position on the hemisphere
        hemisphere_random_pos, angles = sample_hemisphere(ray_hemisphere)

        # Move robot to random position on the hemisphere
        robot_sphere_wrld = np.array(initial_pos) + np.array(hemisphere_random_pos)
        robot = robot_touch_spherical(robot, robot_sphere_wrld, initial_pos, angles)

        # If not contact points, continue. 
        if robot.results_at_touch_wrld is None:
            continue
        
        # Filter points with information about contact, make sure there are at least 500 valid ones
        filtered_full_pointcloud = utils_raycasting.filter_point_cloud(robot.results_at_touch_wrld)
        if filtered_full_pointcloud.shape[0] < 500:
            print('Point cloud shape is too small')
            continue

        # Sample 500 random points among the valid ones and store them in the data dictionary
        random_indices = np.random.choice(filtered_full_pointcloud.shape[0], 500)
        sampled_pointcloud_wrld = filtered_full_pointcloud[random_indices]
        tcp_pos_wrld, tcp_rpy_wrld, _, _, _ = robot.arm.get_current_TCP_pos_vel_worldframe()
        sampled_pointcloud_wrld = sampled_pointcloud_wrld - tcp_pos_wrld
        sampled_pointcloud_wrk = utils_mesh.rotate_pointcloud_inverse(sampled_pointcloud_wrld, tcp_rpy_wrld)
        sampled_pointcloud_wrk = sampled_pointcloud_wrk[None, :, :]  # increase dim for stacking
        data['pointclouds'] = np.vstack((data['pointclouds'], sampled_pointcloud_wrk))

        # Full pointcloud to 25 vertices. By default, vertices are converted to workframe.
        verts_wrk = utils_raycasting.pointcloud_to_vertices_wrk(filtered_full_pointcloud, robot, args)
        if verts_wrk.shape[0] != 25:
            print('Mesh does not have 25 vertices or faces not found')
            continue
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

        data['initial_pos'] = np.vstack((data['initial_pos'], initial_pos))

        save_touch_charts(data)
  
    return data


def save_touch_charts(data):
    """
    Receives a dictionary of data and stores it.
    The dictionary contains the following keys:
        - mesh_list = list containing open3d.geometry.TriangleMesh (25 vertices and faces of the local geometry at touch site)
        - tactile_imgs = list of tactile images, np.array(1, 256, 256)
        - pointcloud_list = list of pointclouds, containing 2000 randomly sampled points that    represent the ground truth to compute the chamfer distance
        - obj_index: index of the object, e.g. camera: 101352
        - rot_M_wrld_list: list of rotation matrices to convert from workframe to worldframe. np.array, shape (n, 3, 3)
        - pos_wrld_list: list of positions of the TCP in worldframe. np.array, shape(n, 3)
        - pos_wrk_list: list of positions of the TCP in workframe. np.array, shape(n, 3)
    Returns:
        - touch_charts_data, dictionary with keys: 'verts', 'faces', 'tactile_imgs', 'pointclouds', 'rot_M_wrld;, 'pos_wrld', 'pos_wrk', 'initial_pos'
            - 'verts': shape (n_samples, 75), ground truth vertices for various samples
            - 'tactile_imgs': shape (n_samples, 1, 256, 256)
            - 'pointclouds': shape (n_samples, 2000, 3), points randomly samples on the touch charts mesh surface.
            - 'rot_M_wrld': 3x3 rotation matrix collected from PyBullet.
            - 'pos_wrld': position of the sensor in world coordinates at touch, collected from PyBullet (robots.coords_at_touch)
            - 'pos_wrk': position of the sensor in world frame collected from PyBullet.
    """
    
    touch_charts_data_dir = os.path.dirname(results.__file__)

    touch_charts_data_path = os.path.join(touch_charts_data_dir, 'touch_charts_gt.npy')

    np.save(touch_charts_data_path, data)


"""Deal with b3Warning regarding missing links in the .URDF (SAPIEN)"""
@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different

def get_ray_hemisphere_old(vertices_wrld, initial_obj_pos):
    # Get maximum coordinates of the vertices in world frame
    max_coords = [ np.amax(vertices_wrld[:,0]), np.amax(vertices_wrld[:,1]), np.amax(vertices_wrld[:,2]) ]

    # Get the radius of the hemisphere
    ray_hemisphere = 1.5 * np.sqrt((max_coords[0] - initial_obj_pos[0])**2 + (max_coords[1] - initial_obj_pos[1])**2 + (max_coords[2] - initial_obj_pos[2])**2)

    return ray_hemisphere


def get_ray_hemisphere(mesh):
    diameter = np.linalg.norm(mesh.bounding_box.bounds[1] - mesh.bounding_box.bounds[0])
    ray_hemisphere = ( .5 * diameter ) * 1.1  # slightly larger than the diameter of the object
    return ray_hemisphere


def sample_along_normals(std_dev, pointcloud, normals, N):
    """It samples N points along the normal in both directions.
    Args:
        std_dev: standard deviation of the normal distribution
        pointcloud: pointcloud, np.array (n, 3)
        normals: normals, np.array (n, 3)
        N: number of points to sample along the normal
    Returns:
        sampled_points: np.array (n * N, 3)
        signed_distance: np.array, standard deviation
    """
    # Tile array vertically 
    pointcloud_tile = np.tile(pointcloud, (N, 1))
    normals_tile = np.tile(normals, (N, 1))

    # Signed distance as sampled standard deviation
    signed_distance = np.random.normal(loc=0, scale=std_dev, size=(pointcloud_tile.shape[0], 1))

    # Sample points along normals
    samples_along_normals_global = signed_distance * normals_tile + pointcloud_tile

    return samples_along_normals_global, signed_distance