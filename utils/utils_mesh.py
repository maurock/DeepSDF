from glob import glob
import numpy as np
import trimesh
import os
from copy import deepcopy
import pybullet as pb
import data.objects as objects

def mesh_from_urdf(filepath):
    """
    Receives path to object index containing the .URDF and verts and faces (both np.array).
    Directory tree:
    - obj_idx  <-- filepath
    |   - textured_objs
    |   |   - ...obj
    |- ...
    """
    total_objs = glob(os.path.join(filepath, 'textured_objs/*.obj'))
    verts = np.array([]).reshape((0,3))
    faces = np.array([]).reshape((0,3))

    mesh_list = []
    for obj_file in total_objs:
        mesh = _as_mesh(trimesh.load(obj_file))
        mesh_list.append(mesh)           
                
    verts_list = [mesh.vertices for mesh in mesh_list]
    faces_list = [mesh.faces for mesh in mesh_list]
    faces_offset = np.cumsum([v.shape[0] for v in verts_list], dtype=np.float32)   # num of faces per mesh
    faces_offset = np.insert(faces_offset, 0, 0)[:-1]            # compute offset for faces, otherwise they all start from 0
    verts = np.vstack(verts_list).astype(np.float32)
    faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)]).astype(np.float32)

    mesh = trimesh.Trimesh(verts, faces)
    return mesh


def _as_mesh(scene_or_mesh):
    # Utils function to get a mesh from a trimesh.Trimesh() or trimesh.scene.Scene()
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def mesh_to_pointcloud(verts, faces, n_samples):
    """
    This method samples n points on a mesh. The number of samples for each face is weighted by its size. 

    Params:
        verts = vertices, np.array(n, 3)
        faces = faces, np.array(m, 3)
        n_samples: number of total samples
    
    Returns:
        pointcloud
    """
    mesh = trimesh.Trimesh(verts, faces)
    pointcloud, _ = trimesh.sample.sample_surface(mesh, n_samples)
    pointcloud = pointcloud.astype(np.float32)
    return pointcloud


def rotate_vertices(vertices, rot=[np.pi / 2, 0, 0]):
    """Rotate vertices by 90 deg around the x-axis. """
    new_verts = deepcopy(vertices)
    # Rotate object
    rot_Q_obj = pb.getQuaternionFromEuler(rot)
    rot_M_obj = np.array(pb.getMatrixFromQuaternion(rot_Q_obj)).reshape(3, 3)
    new_verts = np.einsum('ij,kj->ik', rot_M_obj, new_verts).transpose(1, 0)
    return new_verts


def save_touch_charts(mesh_list, tactile_imgs, pointcloud_list, rot_M_wrld_list, pos_wrld_list, pos_wrk_list, initial_pos, path):
    """
    Receive list containing open3D.TriangleMesh of the local touch charts (25 vertices) and tactile images related to those meshes. It saves a dictionary containing vertices and faces as np array, and normalised tactile images. 

    Parameters:
        - mesh_list = list containing open3d.geometry.TriangleMesh (25 vertices and faces of the local geometry at touch site).
                        len (num touches)
        - tactile_imgs = np.array of tactile images, shape (num_touches, 256, 256)
        - pointcloud_list = np.array of pointclouds, containing 2000 randomly sampled points that represent the ground truth to compute the chamfer distance, shape (num_touches, 2000, 3)
        - rot_M_wrld_list: np.array of rotation matrices to convert from workframe to worldframe. shape (num_touches, 3, 3)
        - pos_wrld_list: np.array of positions of the TCP in worldframe. shape(num_touches, 3)
        - pos_wrk_list: np.array of positions of the TCP in workframe. shape(n, 3)
        - initial_pos: list of initial obj pos, len (3)
    Returns:
        - touch_charts_data, dictionary with keys: 'verts', 'faces', 'tactile_imgs', 'pointclouds', 'rot_M_wrld;, 'pos_wrld', 'pos_wrk'
            - 'verts': shape (n_samples, 75), ground truth vertices for various samples
            - 'faces': shape (n_faces, 3), concatenated triangles. The number of faces per sample varies, so it is not possible to store faces per sample.
            - 'tactile_imgs': shape (n_samples, 1, 256, 256)
            - 'pointclouds': shape (n_samples, 2000, 3), points randomly samples on the touch charts mesh surface.
            - 'rot_M_wrld': 3x3 rotation matrix collected from PyBullet.
            - 'pos_wrld': position of the sensor in world coordinates at touch, collected from PyBullet (robots.coords_at_touch)
            - 'pos_wrk': position of the sensor in world frame collected from PyBullet.
    """
    verts = np.array([], dtype=np.float32).reshape(0, 75)
    faces = np.array([], dtype=np.float32).reshape(0, 3)
    touch_charts_data = dict()

    for mesh in mesh_list:
        vert = np.asarray(mesh.vertices, dtype=np.float32).ravel()
        verts = np.vstack((verts, vert))
        faces = np.vstack((faces, np.asarray(mesh.triangles, dtype=np.float32)))   # (n, 3) not possible (b, n, 3) because n is not constant

    touch_charts_data['verts'] = verts
    touch_charts_data['faces'] = faces

    # Conv2D requires [batch, channels, size1, size2] as input. tactile_imgs is currently [num_samples, size1, size2]. I need to add a second dimension.
    tactile_imgs = np.expand_dims(tactile_imgs, 1) / 255     # normalize tactile images
    touch_charts_data['tactile_imgs'] = tactile_imgs
    touch_charts_data['pointclouds'] = pointcloud_list

    # Store data for rotation and translation
    touch_charts_data['rot_M_wrld'] = rot_M_wrld_list
    touch_charts_data['pos_wrld'] = pos_wrld_list
    touch_charts_data['pos_wrk'] = pos_wrk_list
    touch_charts_data['initial_pos'] = initial_pos

    np.save(path, touch_charts_data)


def get_mesh_z(obj_index, scale):
    """
    Compute the mesh geometry and return the initial z-axis. This is to avoid that the object
    goes partially throught the ground.
    """
    filepath_obj = os.path.join(os.path.dirname(objects.__file__), obj_index)
    mesh = mesh_from_urdf(filepath_obj)
    verts = mesh.vertices
    pointcloud_s = scale_pointcloud(np.array(verts), scale)
    pointcloud_s_r = rotate_pointcloud(pointcloud_s)
    z_values = pointcloud_s_r[:, 2]
    height = (np.amax(z_values) - np.amin(z_values))
    return height/2


def scale_pointcloud(pointcloud, scale=0.1):
    obj = deepcopy(pointcloud)
    obj = obj * scale
    return obj


def rotate_pointcloud(pointcloud, rot=[np.pi / 2, 0, 0]):
    """
    The default rotation reflects the rotation used for the object during data collection
    """
    obj = deepcopy(pointcloud)
    # Rotate object
    rot_Q_obj = pb.getQuaternionFromEuler(rot)
    rot_M_obj = np.array(pb.getMatrixFromQuaternion(rot_Q_obj)).reshape(3, 3)
    obj = np.einsum('ij,kj->ik', rot_M_obj, obj).transpose(1, 0)
    return obj


def get_ratio_urdf_deepsdf(mesh_urdf):
    """Get the ratio between the mesh in the URDF file and the processed DeepSDF mesh."""
    vertices = mesh_urdf.vertices - mesh_urdf.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    max_distances = np.max(distances)  # this is the ratio as well

    return max_distances


def preprocess_urdf():
    """The URDF mesh is processed by the loadURDF method in pybullet. It is scaled and rotated.
    This function achieves the same purpose: given a scale and a rotation matrix or quaternion, 
    it returns the vertices of the rotated and scaled mesh."""
    pass


def debug_draw_vertices_on_pb(vertices_wrld, color=[235, 52, 52]):
    color = np.array(color)/255
    color_From_array = np.full(shape=vertices_wrld.shape, fill_value=color)
    pb.addUserDebugPoints(
        pointPositions=vertices_wrld,
        pointColorsRGB=color_From_array,
        pointSize=1
    )