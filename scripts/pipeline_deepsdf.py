import numpy as np
import os
from utils import utils_mesh, utils_deepsdf
import argparse
from model import model_sdf
import torch
import data
from results import runs_sdf, runs_touch_sdf
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
from glob import glob
from scripts import extract_checkpoints_touch_sdf
import trimesh
from pytorch3d.loss import chamfer_distance
import random 
from utils.utils_metrics import earth_mover_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Second step of the pipeline: predict the object shape from the touch data"""

#@profile
def main(args):
    # Logging
    test_dir = os.path.join(os.path.dirname(runs_touch_sdf.__file__), args.folder_touch_sdf, f"infer_latent_{datetime.now().strftime('%d_%m_%H%M%S')}_{random.randint(0, 10000)}")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    log_path = os.path.join(test_dir, 'settings.txt')
    args_dict = vars(args)  # convert args to dict to write them as json
    with open(log_path, mode='a') as log:
        log.write('Settings:\n')
        log.write(json.dumps(args_dict).replace(', ', ',\n'))
        log.write('\n\n')

    # Load sdf model
    sdf_model = model_sdf.SDFModelMulti(
        num_layers=8, 
        no_skip_connections=False,
        inner_dim=args.inner_dim,
        latent_size=args.latent_size,
        positional_encoding_embeddings=args.positional_encoding_embeddings).float().to(device)
    
    # Load weights for sdf model
    weights_path = os.path.join(os.path.dirname(runs_sdf.__file__), args.folder_sdf, 'weights.pt')
    sdf_model.load_state_dict(torch.load(weights_path, map_location=device))
    sdf_model.eval()
    
    # Initial verts of the default touch chart
    chart_location = os.path.join(os.path.dirname(data.__file__), 'touch_chart.obj')
    initial_verts, initial_faces = utils_mesh.load_mesh_touch(chart_location)
    initial_verts = torch.unsqueeze(initial_verts, 0)

    # Instantiate grid coordinates for mesh extraction
    coords, grid_size_axis = utils_deepsdf.get_volume_coords(args.resolution)
    coords = coords.clone().to(device)
    coords_batches = torch.split(coords, 500000)

    # Save checkpoint
    checkpoint_dict = dict()
    checkpoint_path = os.path.join(test_dir, 'checkpoint_dict.npy')

    # Get the average optimised latent code
    results_sdf_path = os.path.join(os.path.dirname(runs_sdf.__file__), args.folder_sdf, 'results.npy')
    results_sdf = np.load(results_sdf_path, allow_pickle=True).item()
    latent_code = results_sdf['train']['best_latent_codes']
    # Get average latent code (across dimensions)
    latent_code = torch.mean(torch.tensor(latent_code, dtype=torch.float32), dim=0).to(device)
    latent_code.requires_grad = True

    data_folders = glob(os.path.join(os.path.dirname(runs_touch_sdf.__file__), args.folder_touch_sdf, 'data', '*/'))

    # Infer latent code
    for data_folder in data_folders:

        # Sample folder to store tensorboard log and inferred latent code
        num_sample = data_folder.split('/')[-2]

        # If we don't want to reconstruct all, only reconstruct the object with the specified number of samples
        if (args.mode_reconstruct == 'fixed') and (int(num_sample) + 1 not in args.num_samples_extraction):
            continue

        sample_dir = os.path.join(test_dir, num_sample)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        writer = SummaryWriter(log_dir=sample_dir)

        # Load pointclouds and sdf ground truth
        points_sdf = torch.load(os.path.join(data_folder, 'points_sdf.pt'), map_location=device)
        pointclouds_deepsdf = points_sdf[0]
        sdf_gt = points_sdf[1]

        # Infer latent code
        best_latent_code = sdf_model.infer_latent_code(args, pointclouds_deepsdf, sdf_gt, writer, latent_code)

        if args.finetuning:
            best_weights = sdf_model.finetune(args, best_latent_code, pointclouds_deepsdf, sdf_gt, writer)
            sdf_model.load_state_dict(best_weights)

        # Extract mesh obtained with the latent code optimised at inference
        sdf = utils_deepsdf.predict_sdf(best_latent_code, coords_batches, sdf_model)
        vertices_deepsdf, faces_deepsdf = utils_deepsdf.extract_mesh(grid_size_axis, sdf)

        # Save mesh, pointclouds, and their signed distance
        checkpoint_dict[num_sample] = dict()
        checkpoint_dict[num_sample]['mesh'] = [vertices_deepsdf, faces_deepsdf]
        checkpoint_dict[num_sample]['pointcloud'] = pointclouds_deepsdf.cpu()
        checkpoint_dict[num_sample]['sdf'] = sdf_gt.cpu()
        checkpoint_dict[num_sample]['latent_code'] = best_latent_code.cpu()
        np.save(checkpoint_path, checkpoint_dict)

        # Compute Chamfer Distance
        # Get original and reconstructed meshes
        original_mesh_path = os.path.join(os.path.dirname(runs_touch_sdf.__file__), args.folder_touch_sdf, 'mesh_deepsdf.obj')
        original_mesh = trimesh.load(original_mesh_path)
        reconstructed_mesh = trimesh.Trimesh(vertices_deepsdf, faces_deepsdf)

        # Sample point cloud from both meshes
        original_pointcloud, _ = trimesh.sample.sample_surface(original_mesh, 2048)
        reconstructed_pointcloud, _ = trimesh.sample.sample_surface(reconstructed_mesh, 2048)
        
        # Get chamfer distance
        cd = chamfer_distance(torch.tensor(np.array([original_pointcloud]), dtype=torch.float32),
                              torch.tensor(np.array([reconstructed_pointcloud]), dtype=torch.float32))[0]
        emd = earth_mover_distance(original_pointcloud, reconstructed_pointcloud)

        # Save results in a txt file
        results_path = os.path.join(test_dir, 'metrics.txt')
        with open(results_path, 'a') as log:
            log.write('Sample: {}, CD: {}\n'.format(num_sample, cd))
            log.write('Sample: {}, EMD: {}\n'.format(num_sample, emd))

    if not args.no_mesh_extraction:
        extract_checkpoints_touch_sdf.main(test_dir, os.path.join(os.path.dirname(runs_touch_sdf.__file__), args.folder_touch_sdf))

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Argument for inference
    parser.add_argument(
        "--folder_touch_sdf", default=0, type=str, help="Folder containing the collected point clouds"
    )
    parser.add_argument(
        "--folder_sdf", default=0, type=str, help="Folder containing the sdf model weights"
    )
    parser.add_argument(
        "--latent_size", default=128, type=int, help="Folder containing the touch model weights"
    )
    parser.add_argument(
        "--optimiser", default='Adam', type=str, help="Choose the optimiser out of [Adam, LBFGS]"
    )
    parser.add_argument(
        "--lr", default=0.00001, type=float, help="Learning rate to infer the latent code"
    )
    parser.add_argument(
        "--lr_scheduler", default=False, action='store_true', help="Learning rate to infer the latent code"
    )
    parser.add_argument(
        "--sigma_regulariser", default=0.01, type=float, help="Regulariser for the loss function"
    )
    parser.add_argument(
        "--lr_multiplier", default=0.5, type=float, help="Learning rate multiplier for the scheduler"
    )
    parser.add_argument(
        "--patience", default=50, type=float, help="Patience for latent code inference"
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs for latent code inference"
    )
    parser.add_argument(
        "--epochs_finetuning", default=100, type=int, help="Number of epochs for latent code inference"
    )
    parser.add_argument(
        "--clamp", default=False, action='store_true', help="Clip the network prediction"
    )
    parser.add_argument(
        "--clamp_value", type=float, default=0.1, help="Value of the clip"
    )
    parser.add_argument(
        "--langevin_noise", type=float, default=0, help="If this value is higher than 0, it adds noise to the latent space after every update."
    )
    parser.add_argument(
        "--resolution", type=int, default=50, help="Resolution of the extracted mesh"
    )
    parser.add_argument(
        "--mode_reconstruct", default='all', type=str, choices=['fixed', 'all'], help="Select 'all' or 'fixed' to choose between reconstructing for all the collected samples, or only for a specified number of samples. E.g. with args.num_samples_extraction=10 and args.mode_reconstruct='fixed', only the file with 10 samples will be used to reconstruct the object. Otherwise, all the files will be used, therefore the script reconstruct up to 10 samples."
    )
    parser.add_argument(
        "--num_samples_extraction", type=int, default=10, nargs='+', help="Number of samples on the objects. It can be a single number or a list of numbers, e.g. 10 20 30."
    )
    parser.add_argument(
        "--no_mesh_extraction", default=False, action='store_true', help="When true, do not extract the resulting mesh as html and obj, as well as the touches point cloud."
    )
    parser.add_argument(
        "--inner_dim", type=int, default=512, help="Inner dimensions of the network"
    )
    parser.add_argument(
        "--positional_encoding_embeddings", type=int, default=0, help="Number of embeddingsto use for positional encoding. If 0, no positional encoding is used."
    )
    parser.add_argument(
        "--finetuning", default=False, action='store_true', help="Finetune the network after latent code inference."
    )
    parser.add_argument(
        "--lr_finetuning", type=float, default=0.0001, help="Learning rate for finetune"
    )

    args = parser.parse_args()

    # args.folder_sdf ='12_08_135223'
    # args.folder_touch_sdf ='14_08_115842_9668' 
    # args.lr_scheduler = True
    # args.epochs = 100
    # args.epochs_finetuning = 3
    # args.lr = 0.0005
    # args.patience = 100 
    # args.resolution = 20 
    # args.num_samples_extraction = [20]
    # args.mode_reconstruct = 'fixed'
    # args.langevin_noise = 0.0
    # args.num_samples = 20
    # args.clamp = 0.05
    # args.clamp = True

    main(args)