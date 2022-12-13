import numpy as np
import trimesh
from utils import utils_mesh

def rotate_vector_by_quaternion(v, q):
    """
    q is the quaternion given by pybullet in the form (x, y, z, w).
    v is the vector to translate. In our case, we want to translate z=[0,0,1],
    which is the normal of the plane on which the TacTip base is placed.
    v_prime is the translated vector
    This is taken from OpenGLM: https://github.com/g-truc/glm/blob/master/glm/detail/type_quat.inl
    Parameters:
        v = vector to rotate
        q = orientation of the TCP (tuple dim 4)
    """
    u = np.array([q[0], q[1], q[2]])
    s = q[3]
    v_prime = v + ((np.cross(u, v) * s) + np.cross(u, np.cross(u, v))) * 2.0
    return v_prime

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


def filter_point_cloud(contact_info):
    """ Receives contact information from PyBullet. It filters out null contact points
    Params:
        contact_info = results of PyBullet pb.rayTestBatch. It is a tuple of objectUniqueId,  linkIndex, hit_fraction, hit_position, hit_normal. Contact info is calculated in robot.blocking_move -> get_contact_points()
    Return:
        filtered point cloud
     """
    # filter out null contacts and convert tuple -> np.array()
    contact_info_non_null = contact_info[contact_info[:,0]!=-1]
    point_cloud = contact_info_non_null[:, 3]
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

    if args.debug_show_mesh_wrk:
        trimesh.points.plot_points(verts_wrk)

    return verts_wrk
    