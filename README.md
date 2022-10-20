# DeepSDF
Implementation of the paper [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation (Park et al.)](https://arxiv.org/abs/1901.05103)

<p align="center">
  <img src="images/single_shape.png" width="900"  >
</p>

# Content
- [Installation](#installation)
- [Data making](#data-making)
- [Run](#touch-prediction)
- [TODO](#todo)

# Installation
The code was tested on macOS Monterey (M1) and Linux CentOS 7.
```
conda create -n deepsdf python=3.8
conda activate deepsdf
git clone https://github.com/maurock/DeepSDF.git
pip install -e .
```
Create the required directories:
```
bash create_directories.sh
```
This repository uses Python 3.8, PyTorch 1.9, skicit-image. Additional instructions to install the required packages will be added soon.

# Data making
In this code, we used the URDF files from the [PartNet-Mobility Dataset](https://sapien.ucsd.edu/downloads).
The folders that you download from the PartNet website (e.g. `3398`, `3517`) need to be stored in `data\objects`:
```
root
 ├── data
 │   ├── objects
 │   │   ├── 3398 # your obj file here  
 │   │   ├── 3517 # your obj file here  
```
Then, you can extract the meshes as:
```
python extract_urdf.py
```
To train a DeepSDF model we need to generate data from our meshes:
```
python extract_data.py --num_samples_on_surface 10000 --num_samples_far_surface 10000
```
where:
- `num_samples_on_surface`: number of points sampled on the surface, Adttitionally, for each point on tbhe surface of the mesh, two points are sampled nearby according to a Gaussian distribution - as described in the original paper.
- `num_samples_far_surface`: number of points sampled far away from the surface. 

# Run
To train ther model for a single shape:
```
python train.py
```
To see the obtained mesh (using marching cubes):
```
python test.py --run_folder 
```
where:
- `run_folder`: folder that contains the weights of the experiment, you can find it in `results\<timestamp_of_the_experiment>`

# TODO
- [ ] Add multiple shapes
