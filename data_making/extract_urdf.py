import os
import data.objects as objects
import results
from utils import utils_mesh
import numpy as np
from copy import deepcopy
from glob import glob
"""
Extract URDFs from the PartNet-Mobility dataset and store vertices and faces in a dictionary.
"""


def normalise_obj(verts):
    """
    Normalise the object so that its largest dimension is bounded between -1 and 1.
    Scale the other dimensions accordingly. Centre the object at the origin.
    """
    verts_norm = deepcopy(verts)

    # Range of each axis (x, y, z)
    coord_range_list = []

    for i in range(3):
        coord_range_list.append(np.amax(verts[:, i]) - np.amin(verts[:, i]))

        coord_centre = (np.amax(verts[:, i]) + np.amin(verts[:, i]))/2

        verts_norm[:, i] = verts[:, i] - coord_centre

    max_range_half = (sorted(coord_range_list)[-1]) / 2

    verts_norm = verts_norm / np.array([max_range_half])
    return verts_norm


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
        mesh = utils_mesh.urdf_to_mesh(filepath_obj)

        verts, faces = np.array(mesh.vertices), np.array(mesh.faces)

        verts_norm = normalise_obj(verts)

        new_verts = utils_mesh.rotate_pointcloud(verts_norm)
        
        objs_dict[obj_index]['verts'] = new_verts
        objs_dict[obj_index]['faces'] = faces
    return objs_dict  


def main():
    obj_dir = os.path.dirname(objects.__file__)
    objs_dict = load_save_objects(obj_dir)
    np.save(os.path.join(os.path.dirname(results.__file__), 'objs_dict.npy'), objs_dict)

if __name__=='__main__':
    main()