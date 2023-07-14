# DeepSDF
Implementation of the paper [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html). The goal if this repository is to provide a simple and intuitive implementation of the DeepSDF model. Step-to-step instructions on data extraction, training, reconstruction and shape completion are also provided.

For the official implementation and citation guidelines, [check here](https://github.com/facebookresearch/DeepSDF).

# Content
- [Installation](#installation)
- [Usage](#usage)
- [Data making](#data-making)
- [Training](#training)
- [Known Issues](#known-issues)

# Installation
The installation instructions reported in this file were tested on macOS Monterey - Apple M1. This repository also runs on Ubuntu and CentOS 7, but dependency issues between PyTorch and PyTorch3D need to be manually resolved.
```
conda create -n deepsdf python=3.11
conda activate deepsdf
conda env update -n deepsdf --file environment.yml
```
Clone the repository and run the following command from the root directory:
```
pip install .
```
Create the required directories:


# Usage
This section provides an example of usage to predict the shape of an object using pretrained models.

To collect touches and predict the geometry of an object, run:
```
python scripts/pipeline_tactile_deepsdf.py --show_gui --num_samples 20 --obj_folder '02942699/6d036fd1c70e5a5849493d905c02fa86' --folder_sdf '24_03_190521' --lr_scheduler --folder_touch '30_05_1633' --mode_reconstruct 'fixed' --epochs 5000 --lr 0.0005 --patience 100 --resolution 256 --lr_multiplier 0.95 --num_samples_extraction 20 --positional_encoding_embeddings 0 --augment_points_std 0.0005 --augment_multiplier_out 5 --clamp --clamp_value 0.1
```
To speed up the simulation press `G`. Collisions between the arm and the object during sampling are expected, as the collision between the two elements is set to False for speed purposes. This does not affect the resulting contact geometry. The script creates a new folder `results/runs_touch_sdf/<FOLDER_RESULT>`. Please note: as some operations (e.g. weight_norm) do not work on the M1 architecture, inference will run on the CPU. As a result, inference is very slow (up to 1hr for 20 touches and 5000 epochs). On a RTX 3090 GPU, inference takes approx. 5 minutes for the same data points. To change the object to reconstruct, please modify the argument `--obj_folder` by choosing among the objects under the folder `ShapeNetCoreV2urdf`. Additional information regarding these objects can be found in the section `Data making`.

The output is stored in the directory shown at the end of the inference procedure, e.g. `08_06_221924_7367/infer_latent_08_06_223240/19/..`:
- `final_mesh.obj`: the predicted mesh.
- `original_touch.html`: interactive plot showing point clouds of the predicted local surfaces and original object mesh.
- `original_final.html`: interactive plot showing point clouds of the predicted object mesh and original object mesh.
- `final_touch.html`: interactive plot showing point clouds of the predicted local surfaces and predicted object mesh.

Additionally, a file containing the predicted chamfer distance `chamfer_distance.txt` is stored in `results/runs_touch_sdf/<FOLDER_RESULT>`.

# Data making
In this repository, we used the objects from the [ShapeNet](https://shapenet.org/) dataset. The ShapeNetCore.V2 dataset only contains non-watertight `.obj` meshes. However, (1) the PyBullet simulator requires `.urdf` meshes and (2) DeepSDF works best on watertight meshes. Therefore, the `obj` files need to be converted into watertight `urdf` and `obj` meshes for this approach to work. The current repository provides a few examples of shapes in their correct format. Due to the double blind review process, libraries used to process additional shapes cannot be linked here. Further instructions will be added once the review process is complete. 
The required data structure for the `urdf` and `obj` watertight meshes is the following:
```
root
 ├── data
 │   ├── ShapeNetCoreV2urdf
 │   │   ├── 02942699 
 |   |   |   ├── 1ab3abb5c090d9b68e940c4e64a94e1e
 |   |   |   |   ├── model.urdf
 |   |   |   |   ├── model.obj
```

SDF values need to be extracted to train the DeepSDF model:
```
python data_making/extract_sdf.py --num_samples_on_surface 20000 --num_samples_in_bbox 10000 --num_samples_in_volume 5000
```

Data necessary to train the local surface prediciton model is extracted as follows:
```
python extract_touch_charts.py --num_samples 10
```

# Training
To train the local surface prediction model:
```
python model/train_touch.py --epochs 1000 --batch_size 16  --loss_coeff 1000 --lr_scheduler --lr_multiplier 0.8
```
The train the DeepSDF model:
```
python model/train_sdf.py --epochs 150 --lr_model 5e-4 --lr_latent 4e-2 --sigma_regulariser 0.01 --num_layers 8 --batch_size 20480 --lr_multiplier 0.9 --patience 5 --lr_scheduler --latent_size 128 --inner_dim 512 --clamp --clamp_value 0.05 --positional_encoding_embeddings 0
```
The batch size needs to be adjusted to the dataset dimension. For 1300 shapes, a batch size of 20480 results in a 15 hours training (RTX 3090).

# Known issues
- If `--show_gui` is not passed as argument on macOS M1, the collected touches are empty. This can be solved by recalibrating the sensors. On Ubuntu and CentOS 7, `show_gui` can be safely set as False to speed up the data collection procedure.
- Set `PYOPENGL_PLATFORM=egl` before running scripts requiring rendering when using Ubuntu. Example: `PYOPENGL_PLATFORM=egl python scripts/pipeline_tactile_deepsdf.py`.
- Compatibility issues between PyTorch and PyTorch3D on Ubuntu. Please refer to `https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md` for instructions.
- CPU inference is very slow.
