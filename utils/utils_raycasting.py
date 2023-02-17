import numpy as np
import trimesh
from utils import utils_mesh
import results
import os

def create_meshgrid_wrld(c, nx=5, ny=5):
    """
    This function takes the local TCP normal (worldframe) and returns the coordinates of n (by default 25)
    points on a grid defined over the plane perpendicular to vector (0,0,1) and passing through the 
    centre of mass. The dimension of this plane are the sensor's width x depth.
    Parameters:
        c = centre of mass
    Returns:
        numpy array (25, 3) representing the locations of the points on the simple grid wrt worldframe
    """
    x = np.linspace(c[0]-0.015, c[0]+0.015, nx)
    y = np.linspace(c[1]-0.015, c[1]+0.015, ny)
    xv, yv = np.meshgrid(x, y)
    z = np.full(nx * ny, c[2])
    grid_vecs = np.dstack((xv.ravel(),yv.ravel(),z))[0]
    return grid_vecs


def grid_to_TCP_wlrd(c, z_TCP_wrld, nx, ny):
    """
    Rotate all the points in the simple grid to the TCP local frame (wrt worldframe)
    Parameters:
        c = centre of mass in worldframe
        q = orientation of the TCP (tuple dim 4)
    Returns
        the position of the points on the simple grid, np.array (25, 3)
        the position of the points on the transformed grid. These points lie on a plane
            perpendicular to the TCP norm and passing through its centre of mass, np.array(25, 3)
    """
    grid_vecs = create_meshgrid_wrld(c, nx, ny)
    grid_vecs_TCP_wrld = grid_vecs + 0.025*z_TCP_wrld
    return grid_vecs, grid_vecs_TCP_wrld


def shoot_rays(current_TCP_pos_vel_worldframe, pb, nx, ny, draw_rays=False):
    """
    Shoot rays from a plane build around the centre of mass. The plane normal is parallel to the 
    TCP normal.
    Parameters:
        current_TCP_pos_vel_worldframe: returned by robot.arm.get_current_TCP_pos_vel_worldframe()
        pb: pybullet lib to draw vectors
    Return
        grid_vecs: the position of the points on the simple grid around centre of mass, np.array (25, 3)
        grid_vecs_TCP_wrld: the position of the points on the transformed grid. These points lie on a plane
            perpendicular to the TCP norm and passing through its centre of mass, np.array(25, 3)
    """
    # multiple vectors parallel to TCP normal
    # simple grid of points to rotate. It's used to shoot a batch of rays to extract the local shape
    z_wrk = np.array([0, 0, 1])

    rpy_wrld = np.array(current_TCP_pos_vel_worldframe[1]) # TCP orientation

    z_TCP_wrld = utils_mesh.rotate_pointcloud(np.array([z_wrk]), rpy_wrld)[0]

    c_wrld = current_TCP_pos_vel_worldframe[0] # TCP centre of mass

    grid_vecs, grid_vecs_TCP_wrld = grid_to_TCP_wlrd(c_wrld, z_TCP_wrld, nx, ny)

    if draw_rays:
        for i in range(0, 25):
            start_point = grid_vecs_TCP_wrld[i] - 0.05 * z_TCP_wrld
            end_point = start_point + 0.025 * z_TCP_wrld
            pb.addUserDebugLine(start_point, end_point, lifeTime=0.05)
    
    return grid_vecs, grid_vecs_TCP_wrld, z_TCP_wrld


def get_contact_points(current_TCP_pos_vel_worldframe, pb, sensor, nx=5, ny=5, draw_points=False):
    """
    When the sensor touches a surface, returns the contact points along with other info. 
    Parameters:
        current_TCP_pos_vel_worldframe: returned by robot.arm.get_current_TCP_pos_vel_worldframe()
        pb: pybullet lib to draw vectors
    Return
        results: objectUniqueId, linkIndex, hit_fraction, hit_position, hit_normal
    """
    if sensor=='tactip':
        # Get static points on the TacTip in the workframe
        points_path = os.path.join(os.path.dirname(results.__file__), 'tactip_contact_points.npy')
        points_on_sensor_wrk = np.load(points_path)

        rpy_wrld = np.array(current_TCP_pos_vel_worldframe[1]) # TCP orientation

        normal_wrk = np.array([0, 0, 1])
        normal_wrld = utils_mesh.rotate_pointcloud(np.array([normal_wrk]), rpy_wrld)[0]
        
        # Get the centre of the tactip
        central_point = current_TCP_pos_vel_worldframe[0] - 0.02 * normal_wrld

        # Convert from workframe to worldframe
        points_on_sensor_wrld = utils_mesh.rotate_pointcloud(points_on_sensor_wrk, rpy_wrld)
        points_on_sensor_wrld = points_on_sensor_wrld + central_point

        raysFrom = np.tile(central_point, (points_on_sensor_wrld.shape[0], 1)) 
        raysTo = points_on_sensor_wrld

        # debug points
        if draw_points:
            color = np.array([235, 52, 52])/255
            color_From_array = np.full(shape=raysTo.shape, fill_value=color)
            pb.addUserDebugPoints(
                pointPositions=raysTo,
                pointColorsRGB=color_From_array,
                pointSize=1
            )

    else:
        _, grid_vecs_TCP_wrld, z_TCP_wrld = shoot_rays(current_TCP_pos_vel_worldframe, pb, nx, ny)
        # the grid is defined 0.5 units in front of the plane passing through the TCP centrer of mass
        raysFrom = grid_vecs_TCP_wrld - 0.05 * z_TCP_wrld  
        # 0.025 is approx. the distance necessary to cover the entire TCP height + a bit more
        raysTo = raysFrom + 0.025 * z_TCP_wrld

    # shoot rays in batches (the max allowed batch), then put them all together
    max_rays = pb.MAX_RAY_INTERSECTION_BATCH_SIZE
    size = raysFrom.shape[0]
    end = 0
    if size > max_rays:
        results_rays = []
        start = 0
        while end != size:
            end = start + max_rays if (start + max_rays < size) else size
            rays = pb.rayTestBatch(raysFrom[start:end], raysTo[start:end], numThreads=0)
            results_rays.append(rays)
            start = start + max_rays
        results_rays = np.array(results_rays, dtype=object)
    else:
        results_rays = np.array(pb.rayTestBatch(raysFrom, raysTo), dtype=object)

    return results_rays


def filter_point_cloud(contact_info, obj_id):
    """ Receives contact information from PyBullet. It filters out null contact points
    Params:
        contact_info = results of PyBullet pb.rayTestBatch. It is a tuple of objectUniqueId,  linkIndex, hit_fraction, hit_position, hit_normal. Contact info is calculated in robot.blocking_move -> get_contact_points()
    Return:
        filtered point cloud
     """
    # filter out points not intersecting with the object and convert tuple -> np.array()
    contact_info_on_obj = contact_info[contact_info[:,0] == obj_id]
    point_cloud = contact_info_on_obj[:, 3]
    point_cloud = np.array([np.array(_) for _ in point_cloud])

    return point_cloud


def pointcloud_to_vertices_wrk(point_cloud, robot, args):
    """
    Method to reduce the point cloud obtained from pyBullet into 25 vertices.
    It receives filtered contact information and computes 25 k_means for the non-null contact points. 
    Vertices are converted to workframe.
    
    Params:
        point_cloud = filtered point cloud, null contact points are not included
    Return:
        mesh = open3d.geometry.TriangleMesh, 25 vertices and faces of the local geometry at touch site
    """
    # compute k-means that will used as vertices
    print(f'Shape of full pointcloud: {point_cloud.shape}')

    verts_wrld = trimesh.points.k_means(point_cloud, 25)[0]

    # P_tcp = R_tcp/wd * P_wd = (R_wd/tcp)^-1 * P_wd
    # where R_wd/tcp = rotation of TCP in worldframe, which is the result of get_current_TCP_pos_vel_worldframe()
    tcp_pos_wrld, tcp_rpy_wrld, _, _, _ = robot.arm.get_current_TCP_pos_vel_worldframe()

    pointcloud_wrld = verts_wrld - tcp_pos_wrld
    verts_wrk = utils_mesh.rotate_pointcloud_inverse(pointcloud_wrld, tcp_rpy_wrld)

    print(f'Point cloud to vertices: {verts_wrk.shape}')

    return verts_wrk
    