import os
import data.objects as objects
import results
from glob import glob
from utils import utils_deepsdf
from utils import utils_mesh
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


def main():
    obj_dir = os.path.dirname(objects.__file__)
    objs_dict = load_save_objects(obj_dir)
    np.save(os.path.join(os.path.dirname(results.__file__), 'objs_dict.npy'), objs_dict)

if __name__=='__main__':
    main()