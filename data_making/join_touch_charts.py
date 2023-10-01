"""Script to join the extracted touch charts"""
import os
import numpy
from glob import glob
import argparse
import numpy as np
import results
import pickle

def main(args):
    touch_charts_dir = os.path.join(os.path.dirname(results.__file__), args.name_folder)

    touch_charts_paths = glob(os.path.join(touch_charts_dir, '*.npy'))

    num_valid_points = 200 # same used in extract_touch_charts.py

    # Initialise dict with arrays to store.
    data = {
        "tactile_imgs": np.array([], dtype=np.float32).reshape(0, 1, 256, 256),
        "pointclouds": np.array([], dtype=np.float32).reshape(0, num_valid_points, 3),   # fixed dimension touch chart pointcloud (workframe)
        "rot_M_wrld_list": np.array([], dtype=np.float32).reshape(0, 3, 3),      # rotation matrix (work wrt worldframe)
        "pos_wrld_list": np.array([]).reshape(0, 3) , # TCP pos (worldframe)
        "pos_wrk_list": np.array([], dtype=np.float32).reshape(0, 3),   # TCP pos (worldframe)
        "obj_index": np.array([], dtype=np.float32).reshape(0, 1),
        "initial_pos": np.array([], dtype=np.float32).reshape(0, 3)
    }

    for touch_charts_path in touch_charts_paths:

        data_temp = np.load(touch_charts_path, allow_pickle=True).item()

        data['pointclouds'] = np.vstack((data['pointclouds'], data_temp['pointclouds']))

        data['pos_wrld_list'] = np.vstack((data['pos_wrld_list'], data_temp['pos_wrld_list']))

        data['pos_wrk_list'] = np.vstack((data['pos_wrk_list'], data_temp['pos_wrk_list']))

        data['rot_M_wrld_list'] = np.vstack((data['rot_M_wrld_list'], data_temp['rot_M_wrld_list']))

        data['obj_index'] = np.vstack((data['obj_index'], data_temp['obj_index']))

        data['initial_pos'] = np.vstack((data['initial_pos'], data_temp['initial_pos']))

        data['tactile_imgs'] = np.vstack((data['tactile_imgs'], data_temp['tactile_imgs']))       

    output_path = f'{touch_charts_dir}.npy'

    with open(f'{touch_charts_dir}.pkl', 'wb') as file:
        pickle.dump(data, file, protocol=4)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_folder", default='', type=str, help="Folder that contains the touch charts to join under results/'"
    )
    args = parser.parse_args()

    main(args)