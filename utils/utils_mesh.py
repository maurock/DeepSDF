from glob import glob
import numpy as np
import trimesh
import os
from copy import deepcopy
import pybullet as pb
import data.objects as objects
from data_making import extract_urdf
import torch
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords
from pytorch3d.loss import chamfer_distance as cuda_cd
from pytorch3d.io.obj_io import load_obj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def urdf_to_mesh(filepath):
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


def mesh_to_pointcloud(mesh, n_samples):
    """
    This method samples n points on a mesh. The number of samples for each face is weighted by its size. 

    Params:
        mesh = trimesh.Trimesh()
        n_samples: number of total samples
    
    Returns:
        pointcloud
    """
    pointcloud, _ = trimesh.sample.sample_surface(mesh, n_samples)
    pointcloud = np.array(pointcloud).astype(np.float32)
    return pointcloud


def rotate_vertices(vertices, rot=[np.pi / 2, 0, 0]):
    """Rotate vertices by 90 deg around the x-axis. """
    new_verts = deepcopy(vertices)
    # Rotate object
    rot_Q_obj = pb.getQuaternionFromEuler(rot)
    rot_M_obj = np.array(pb.getMatrixFromQuaternion(rot_Q_obj)).reshape(3, 3)
    new_verts = np.einsum('ij,kj->ik', rot_M_obj, new_verts).transpose(1, 0)
    return new_verts


def calculate_initial_z(obj_index, scale):
    """
    Compute the mesh geometry and return the initial z-axis. This is to avoid that the object
    goes partially throught the ground.
    """
    filepath_obj = os.path.join(os.path.dirname(objects.__file__), obj_index)
    mesh = urdf_to_mesh(filepath_obj)
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


def rotate_pointcloud(pointcloud_A, rpy_BA=[np.pi / 2, 0, 0]):
    """
    The default rotation reflects the rotation used for the object during data collection.
    This calculates P_b, where P_b = R_b/a * P_a.
    R_b/a is rotation matrix of a wrt b frame.
    """
    # Rotate object
    rot_Q = pb.getQuaternionFromEuler(rpy_BA)
    rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)
    pointcloud_B = np.einsum('ij,kj->ik', rot_M, pointcloud_A).transpose(1, 0)

    return pointcloud_B


def rotate_pointcloud_inverse(pointcloud_A, rpy_AB):
    """
    This calculates P_b, where P_b = (R_a/b)^-1 * P_a.
    R_b/a is rotation matrix of a wrt b frame."""
    rot_Q = pb.getQuaternionFromEuler(rpy_AB)
    rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)
    rot_M_inv = np.linalg.inv(rot_M)
    pointcloud_B = rot_M_inv @ pointcloud_A.transpose(1,0)
    pointcloud_B = pointcloud_B.transpose(1,0)

    return pointcloud_B
    

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


def translate_rotate_mesh(pos_wrld_list, rot_M_wrld_list, pointclouds_list, obj_initial_pos):
    """
    Given a pointcloud (workframe), the position of the TCP (worldframe), the rotation matrix (worldframe), it returns the pointcloud in worldframe. It assumes a known position of the object.

    Params:
        pos_wrld_list: (m, 3)
        rot_M_wrld_list: (m, 3, 3)
        pointclouds_list: pointcloud in workframe (m, number_points, 3)
        obj_initial_pos: (3,)

    Returns:
        pointcloud_wrld: (m, number_points, 3)
    """
    a = rot_M_wrld_list @ pointclouds_list.transpose(0,2,1)
    b = a.transpose(0,2,1)
    c = pos_wrld_list[:, np.newaxis, :] + b
    pointcloud_wrld = c - obj_initial_pos
    return pointcloud_wrld


def load_save_objects(obj_dir):
    """
    Extract objects (verts and faces) from the URDF files in the PartNet-Mobility dataset.
    Store objects in dictionaries, where key=obj_idx and value=np.array[verts, faces]

    Args:
        obj_dir: directory containing the object folders
    Returns:
        dictionary of dictionaries, the first key is the object indices, the second
        key are 'verts' and 'faces', both stores as np.array
    """
    # List all the objects in data/objects/
    list_objects = [filepath.split('/')[-2] for filepath in glob(os.path.join(obj_dir, '*/'))]

    objs_dict = dict()
    
    for obj_index in list_objects:

        objs_dict[obj_index] = dict()

        filepath_obj = os.path.join(obj_dir, obj_index)
        mesh = urdf_to_mesh(filepath_obj)

        verts, faces = np.array(mesh.vertices), np.array(mesh.faces)

        verts_norm = extract_urdf.normalise_obj(verts)

        new_verts = rotate_pointcloud(verts_norm)
        
        objs_dict[obj_index]['verts'] = new_verts
        objs_dict[obj_index]['faces'] = faces
    return objs_dict  


# adapted from: https://github.com/facebookresearch/Active-3D-Vision-and-Touch/blob/main/pterotactyl/utility/utils.py
# returns the chamfer distance between a mesh and a point cloud (Ed. Smith)
def chamfer_distance(verts, faces, gt_points, num=1000, repeat=1):
    pred_points= batch_sample(verts, faces, num=num)
    cd, _ = cuda_cd(pred_points, gt_points, batch_reduction=None)
    if repeat > 1:
        cds = [cd]
        for i in range(repeat - 1):
            pred_points = batch_sample(verts, faces, num=num)
            cd, _ = cuda_cd(pred_points, gt_points, batch_reduction=None)
            cds.append(cd)
        cds = torch.stack(cds)
        cd = cds.mean(dim=0)
    return cd


# implemented from: https://github.com/facebookresearch/Active-3D-Vision-and-Touch/blob/main/pterotactyl/utility/utils.py, MIT License
 # sample points from a batch of meshes
def batch_sample(verts, faces, num=10000):
    # Pytorch3D based code
    bs = verts.shape[0]
    face_dim = faces.shape[0]
    vert_dim = verts.shape[1]
    # following pytorch3D convention shift faces to correctly index flatten vertices
    F = faces.unsqueeze(0).repeat(bs, 1, 1)
    F += vert_dim * torch.arange(0, bs).unsqueeze(-1).unsqueeze(-1).to(F.device)
    # flatten vertices and faces
    F = F.reshape(-1, 3)
    V = verts.reshape(-1, 3)
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(V, F)
        Ar = areas.reshape(bs, -1)
        Ar[Ar != Ar] = 0
        Ar = torch.abs(Ar / Ar.sum(1).unsqueeze(1))
        Ar[Ar != Ar] = 1

        sample_face_idxs = Ar.multinomial(num, replacement=True)
        sample_face_idxs += face_dim * torch.arange(0, bs).unsqueeze(-1).to(Ar.device)

    # Get the vertex coordinates of the sampled faces.
    face_verts = V[F]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(bs, num, V.dtype, V.device)

    # Use the barycentric coords to get a point on each sampled face.
    A = v0[sample_face_idxs]  # (N, num_samples, 3)
    B = v1[sample_face_idxs]
    C = v2[sample_face_idxs]
    samples = w0[:, :, None] * A + w1[:, :, None] * B + w2[:, :, None] * C

    return samples


# implemented from: https://github.com/facebookresearch/Active-3D-Vision-and-Touch/blob/main/pterotactyl/utility/utils.py, MIT License
# loads the initial mesh and returns vertex, and face information
def load_mesh_touch(obj):
    obj_info = load_obj(obj)
    verts = obj_info[0]
    faces = obj_info[1].verts_idx
    verts = torch.FloatTensor(verts).to(device)
    faces = torch.LongTensor(faces).to(device)
    return verts, faces