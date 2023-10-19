import numpy as np
import os
from utils import utils_mesh, utils_deepsdf
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
import yaml
import config_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Second step of the pipeline: predict the object shape from the touch data"""


# @profile
def main(args):
    # Logging
    test_dir = os.path.join(
        os.path.dirname(runs_touch_sdf.__file__),
        args["folder_touch_sdf"],
        f"infer_latent_{datetime.now().strftime('%d_%m_%H%M%S')}_{random.randint(0, 10000)}",
    )
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    log_path = os.path.join(test_dir, "settings.yaml")
    with open(log_path, "w") as f:
        yaml.dump(args, f)

    # Load sdf model
    sdf_model = (
        model_sdf.SDFModelMulti(
            num_layers=8,
            no_skip_connections=False,
            inner_dim=args["inner_dim"],
            latent_size=args["latent_size"],
            positional_encoding_embeddings=args["positional_encoding_embeddings"],
        )
        .float()
        .to(device)
    )

    # Load weights for sdf model
    weights_path = os.path.join(
        os.path.dirname(runs_sdf.__file__), args["folder_sdf"], "weights.pt"
    )
    sdf_model.load_state_dict(torch.load(weights_path, map_location=device))
    sdf_model.eval()

    # Initial verts of the default touch chart
    chart_location = os.path.join(os.path.dirname(data.__file__), "touch_chart.obj")
    initial_verts, initial_faces = utils_mesh.load_mesh_touch(chart_location)
    initial_verts = torch.unsqueeze(initial_verts, 0)

    # Instantiate grid coordinates for mesh extraction
    coords, grid_size_axis = utils_deepsdf.get_volume_coords(args["resolution"])
    coords = coords.clone().to(device)
    coords_batches = torch.split(coords, 500000)

    # Save checkpoint
    checkpoint_dict = dict()
    checkpoint_path = os.path.join(test_dir, "checkpoint_dict.npy")

    # Get the average optimised latent code
    results_sdf_path = os.path.join(
        os.path.dirname(runs_sdf.__file__), args["folder_sdf"], "results.npy"
    )
    results_sdf = np.load(results_sdf_path, allow_pickle=True).item()
    latent_code = results_sdf["train"]["best_latent_codes"]
    # Get average latent code (across dimensions)
    latent_code = torch.mean(torch.tensor(latent_code, dtype=torch.float32), dim=0).to(
        device
    )
    latent_code.requires_grad = True

    data_folders = glob(
        os.path.join(
            os.path.dirname(runs_touch_sdf.__file__),
            args["folder_touch_sdf"],
            "data",
            "*/",
        )
    )

    # Infer latent code
    for data_folder in data_folders:
        # Sample folder to store tensorboard log and inferred latent code
        num_sample = data_folder.split("/")[-2]

        # If we don't want to reconstruct all, only reconstruct the object with the specified number of samples
        if (args["mode_reconstruct"] == "fixed") and (
            int(num_sample) + 1 not in args["num_samples_extraction"]
        ):
            continue

        sample_dir = os.path.join(test_dir, num_sample)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        writer = SummaryWriter(log_dir=sample_dir)

        # Load pointclouds and sdf ground truth
        points_sdf = torch.load(
            os.path.join(data_folder, "points_sdf.pt"), map_location=device
        )
        pointclouds_deepsdf = points_sdf[0]
        sdf_gt = points_sdf[1]

        # Infer latent code
        best_latent_code = sdf_model.infer_latent_code(
            args, pointclouds_deepsdf, sdf_gt, writer, latent_code
        )

        if args["finetuning"]:
            best_weights = sdf_model.finetune(
                args, best_latent_code, pointclouds_deepsdf, sdf_gt, writer
            )
            sdf_model.load_state_dict(best_weights)

        # Extract mesh obtained with the latent code optimised at inference
        sdf = utils_deepsdf.predict_sdf(best_latent_code, coords_batches, sdf_model)
        vertices_deepsdf, faces_deepsdf = utils_deepsdf.extract_mesh(
            grid_size_axis, sdf
        )

        # Save mesh, pointclouds, and their signed distance
        checkpoint_dict[num_sample] = dict()
        checkpoint_dict[num_sample]["mesh"] = [vertices_deepsdf, faces_deepsdf]
        checkpoint_dict[num_sample]["pointcloud"] = pointclouds_deepsdf.cpu()
        checkpoint_dict[num_sample]["sdf"] = sdf_gt.cpu()
        checkpoint_dict[num_sample]["latent_code"] = best_latent_code.cpu()
        np.save(checkpoint_path, checkpoint_dict)

        # Compute Chamfer Distance
        # Get original and reconstructed meshes
        original_mesh_path = os.path.join(
            os.path.dirname(runs_touch_sdf.__file__),
            args["folder_touch_sdf"],
            "mesh_deepsdf.obj",
        )
        original_mesh = trimesh.load(original_mesh_path)
        reconstructed_mesh = trimesh.Trimesh(vertices_deepsdf, faces_deepsdf)

        # Sample point cloud from both meshes
        original_pointcloud, _ = trimesh.sample.sample_surface(original_mesh, 2048)
        reconstructed_pointcloud, _ = trimesh.sample.sample_surface(
            reconstructed_mesh, 2048
        )

        # Get chamfer distance
        cd = chamfer_distance(
            torch.tensor(np.array([original_pointcloud]), dtype=torch.float32),
            torch.tensor(np.array([reconstructed_pointcloud]), dtype=torch.float32),
        )[0]
        emd = earth_mover_distance(original_pointcloud, reconstructed_pointcloud)

        # Save results in a txt file
        results_path = os.path.join(test_dir, "metrics.txt")
        with open(results_path, "a") as log:
            log.write("Sample: {}, CD: {}\n".format(num_sample, cd))
            log.write("Sample: {}, EMD: {}\n".format(num_sample, emd))

    if not args["no_mesh_extraction"]:
        extract_checkpoints_touch_sdf.main(
            test_dir,
            os.path.join(
                os.path.dirname(runs_touch_sdf.__file__), args["folder_touch_sdf"]
            ),
        )


if __name__ == "__main__":
    cfg_path = os.path.join(
        os.path.dirname(config_files.__file__), "pipeline_deepsdf.yaml"
    )
    with open(cfg_path, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)
