import os
import shutil
import data.ShapeNetCoreV2_ALL as ShapeNetCoreV2_ALL
import numpy as np
from tqdm import tqdm
from glob import glob
"""Script to choose the data from ShapeNetCoreV2_ALL and move them to ShapeNetCoreV2urdf"""

if __name__=='__main__':
    # Set directories
    data_dir = os.path.dirname(ShapeNetCoreV2_ALL.__file__)

    output_dir = os.path.join(os.path.dirname(ShapeNetCoreV2_ALL.__file__), '..', 'ShapeNetCoreV2')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Set the categories
    categories = ['bottle', 'mug', 'bowl', 'camera', 'guitar', 'jar']
    categories_dict = {'bottle': '02876657', 'mug': '03797390', 'bowl': '02880940', 'camera': '02942699', 'jar': '03593526', 'guitar': '03467517'}

    # Select 200 folders inside each category from ShapeNetCoreV2_ALL. If the numbers of folder are less than 200, select all the folders
    for category in tqdm(categories):
        # Get the list of all the models, excluding the files that start with '.' e.g. .DS_Store
        models = [a.split(os.sep)[-1] for a in glob(os.path.join(data_dir, categories_dict[category], '[!.]*'))]
        print(f'Category {category}: {len(models)} models')

        # If there are more than 200 models, select 200 random folder
        num_models = len(models) if len(models) <= 300 else 300
        if len(models) > num_models:
            models = np.random.choice(models, num_models, replace=False)

        # For each model
        for model in models:
            # Get the list of all the files
            files = os.listdir(os.path.join(data_dir, categories_dict[category], model, 'models'))

            # If there is a urdf file
            if 'model_normalized.obj' in files:
                # Copy the model to the output directory
                shutil.copytree(os.path.join(data_dir, categories_dict[category], model), os.path.join(output_dir, categories_dict[category], model))