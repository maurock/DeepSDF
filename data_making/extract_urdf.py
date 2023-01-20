import os
import data.objects as objects
import data.ShapeNetCoreV2urdf as ShapeNetCoreV2
import results
from utils import utils_mesh
import numpy as np
from copy import deepcopy
from glob import glob
import argparse
"""
Extract URDFs from the PartNetMobility or ShapeNetCore dataset and store vertices and faces in a dictionary.
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


def load_objects(dataset):
    """
    Extract objects (verts and faces) from the URDF files in the PartNetMobility dataset.
    Store objects in dictionaries, where key=obj_idx and value=np.array[verts, faces]

    Args:
        obj_dir: directory containing the object folders
    Returns:
        dictionary of dictionaries, the first key is the object indices, the second
        key are 'verts' and 'faces', both stores as np.array
    """
    if dataset=='PartNetMobility':

        obj_dir = os.path.dirname(objects.__file__)
        # List all the objects
        list_objects = [filepath.split('/')[-2] for filepath in glob(os.path.join(obj_dir, '*/'))]
        if '__pycache__' in list_objects:
            list_objects.remove('__pycache__')

    elif dataset=='ShapeNetCore':

        obj_dir = os.path.dirname(ShapeNetCoreV2.__file__)
        # List all the objects
        list_objects_split = [filepath.split('/')[-3:-1] for filepath in glob(os.path.join(obj_dir, '*/*/'))]
        list_objects = ['/'.join(i) for i in list_objects_split]

    objs_dict = dict()
    
    for obj_index in list_objects:

        objs_dict[obj_index] = dict()

        obj_dir = os.path.join(obj_dir, obj_index)   # directory to the object folder
        obj_path = os.path.join(obj_dir, 'model.obj')   # path to the URDF file

        verts, faces = np.array(mesh.vertices).astype(np.float16), np.array(mesh.faces).astype(np.float16)
        
        if dataset=='PartNetMobility':

            # Normalise and rotate point clouds
            verts = normalise_obj(verts)
            verts = utils_mesh.rotate_pointcloud(verts)
        
        objs_dict[obj_index]['verts'] = verts
        objs_dict[obj_index]['faces'] = faces

    return objs_dict  


def main(args):
    objs_dict = load_objects(args.dataset)

    np.save(os.path.join(os.path.dirname(results.__file__), f'objs_dict_{args.dataset}.npy'), objs_dict)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default='ShapeNetCore', type=str, help="Dataset used: 'ShapeNetCore' or 'PartNetMobility'"
    )
    args = parser.parse_args()

    main(args)