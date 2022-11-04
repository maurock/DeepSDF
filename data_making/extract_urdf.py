import os
import data.objects as objects
import results
from glob import glob
import utils.utils as utils
import numpy as np
from copy import deepcopy
"""
Extract URDFs from the PartNet-Mobility dataset and store vertices and faces in a dictionary.
"""
def normalise_obj(verts):
    """
    Normalise the object so that it fits in a unit sphere. 
    """
    verts_norm = deepcopy(verts)
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
    Extract objects (verts and faces) from the URDF files in the PartNet-Mobnility dataset.
    Store objects in dictionaries, where key=obj_idx and value=np.array[verts, faces]

    Args:
        obj_dir: directory containing the object folders
    Returns:
        dictionary of dictionaries, the first key is the object indexes, the second
        key are 'verts' and 'faces', both stores as np.array
    """
    # List all the objects in data/objects/
    list_objects = [filepath.split('/')[-1] for filepath in glob(os.path.join(obj_dir, '*'))]
    list_objects.remove('__init__.py')
    if '__pycache__' in list_objects:
        list_objects.remove('__pycache__')
    objs_dict = dict()
    for obj_index in list_objects:
        objs_dict[obj_index] = dict()
        filepath_obj = os.path.join(obj_dir, obj_index)
        verts, faces = utils.mesh_from_urdf(filepath_obj)
        verts_norm = normalise_obj(verts)
        new_verts = utils.rotate_vertices(verts_norm)
        objs_dict[obj_index]['verts'] = new_verts
        objs_dict[obj_index]['faces'] = faces
    return objs_dict  

def main():
    obj_dir = os.path.dirname(objects.__file__)
    objs_dict = load_save_objects(obj_dir)
    np.save(os.path.join(os.path.dirname(results.__file__), 'objs_dict.npy'), objs_dict)

if __name__=='__main__':
    main()