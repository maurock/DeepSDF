import numpy as np
import torch
import results
import os
import meshplot as mp
mp.offline()
import model.sdf_model as sdf_model
import argparse
from tqdm import tqdm 
import utils.utils as utils
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialise_latent_code(latent_size, results_dict):
    """Initialise latent code as the average over all the obtained latent codes"""
    latent_code = torch.mean(torch.from_numpy(results_dict['train']['latent_codes'][-1]), dim=0).view(1, -1).to(device)
    latent_code.requires_grad = True
    #latent_code = torch.normal(0, 0.1, size = (1, latent_size), dtype=torch.float32, requires_grad=True, device=device)
    return latent_code

def get_data(args):
    """Return x and y. Sample N points from the desired object.
    
    Params:
        - idx_obj_dict: index of the object in the dictionary. e.g. 0, 1, 2.. the first, second, third object in objs_dict.
                       If None, a random object is selected.
    
    Returns:
        - coords: points samples on the object
        - sdf_gt: ground truth for the signed distance function 
    """
    # dictionaries
    objs_dict = np.load(os.path.join(os.path.dirname(results.__file__), 'objs_dict.npy'), allow_pickle=True).item()
    samples_dict = np.load(os.path.join(os.path.dirname(results.__file__), 'samples_dict.npy'), allow_pickle=True).item()

    # list of objects
    objs = list(samples_dict.keys())

    # select random object
    if args.index_objs_dict != -1 and args.index_objs_dict < len(objs):
        random_obj = objs[args.index_objs_dict]
    else:
        random_obj = objs[np.random.randint(0, len(objs))]

    # mesh for random object
    mesh = objs_dict[random_obj]

    # sample point cloud on random object
    coords = utils.mesh_to_pointcloud(mesh['verts'], mesh['faces'], args.num_samples)
    coords = torch.from_numpy(coords).to(device)

    sdf_gt = torch.full(size=(coords.shape[0], 1), fill_value=0).to(device)

    return coords, sdf_gt


def main(args):
    folder = args.folder

    # Logging
    test_path = os.path.join(os.path.dirname(results.__file__), 'runs', folder, 'test')
    writer = SummaryWriter(log_dir=test_path)

    model = sdf_model.SDFModelMulti(num_layers=8, no_skip_connections=False).to(device)

    # Load weights
    weights_path = os.path.join(os.path.dirname(results.__file__), 'runs', folder, 'weights.pt')
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Load results dictionary
    results_dict_path = os.path.join(os.path.dirname(results.__file__), 'runs', folder, 'results.npy')
    results_dict = np.load(results_dict_path, allow_pickle=True).item()

    # Initialise latent code and optimiser
    latent_code = initialise_latent_code(args.latent_size, results_dict)
    optim = torch.optim.Adam([latent_code], lr=args.lr)

    if args.lr_scheduler:
        scheduler_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', 
                                                factor=args.lr_multiplier, 
                                                patience=args.patience, 
                                                threshold=0.005, threshold_mode='abs')
    
    # create dataset
    coords, sdf_gt = get_data(args)

    # prediction
    for epoch in tqdm(range(0, args.epochs)):
        latent_code_tile = torch.tile(latent_code, (coords.shape[0], 1))
        x = torch.hstack((latent_code_tile, coords))
        optim.zero_grad()
        predictions = model(x)
        loss_value = utils.SDFLoss_multishape(sdf_gt, predictions, x[:, :args.latent_size], sigma=args.sigma_regulariser)
        loss_value.backward()
        optim.step()
        writer.add_scalar('Training loss', loss_value.detach().cpu().item(), epoch)

        # step scheduler and store on tensorboard
        if args.lr_scheduler:
            scheduler_latent.step(loss_value.item())
            writer.add_scalar('Learning rate', scheduler_latent._last_lr[0], epoch)

        # store latent codes and their gradient on tensorboard
        tag = f"latent_code_0"
        writer.add_histogram(tag, latent_code, global_step=epoch)
        tag = f"grad_latent_code_0"
        writer.add_histogram(tag, latent_code.grad, global_step=epoch)


    # Extract mesh obtained with the latent code optimised at inference
    coords, grad_size_axis = utils.get_volume_coords(args.resolution)

    sdf = utils.predict_sdf(latent_code, coords, model)
    vertices, faces = utils.extract_mesh(grad_size_axis, sdf)

    # save mesh using meshplot
    mesh_path = os.path.join(test_path, f'mesh.html')
    utils.save_meshplot(vertices, faces, mesh_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, default='', help="Folder that contains the network parameters"
    )
    parser.add_argument(
        "--latent_size", type=int, default=128, help="Size of the latent code"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Initial learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=3000, help="Number of epochs"
    )    
    parser.add_argument(
        "--sigma_regulariser", type=float, default=0.01, help="Sigma value for the regulariser in the loss function"
    )
    parser.add_argument(
        "--lr_multiplier", type=float, default=0.5, help="Multiplier for the learning rate scheduling"
    )  
    parser.add_argument(
        "--patience", type=int, default=20, help="Patience for the learning rate scheduling"
    )  
    parser.add_argument(
        "--lr_scheduler", default=False, action='store_true', help="Turn on lr_scheduler"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Num network layers"
    )    
    parser.add_argument(
        "--no_skip_connections", default=False, action='store_true', help="Do not skip connections"
    ) 
    parser.add_argument(
        "--num_samples", type=int, default=5000, help="Number of points to sample on the object surface"
    )    
    parser.add_argument(
        "--index_objs_dict", type=int, default=-1, help="Index of the object in the dictionary. Set this higher than -1 to sample from a specific object"
    )  
    parser.add_argument(
        "--resolution", type=int, default=50, help="Folder that contains the network parameters"
    )
    args = parser.parse_args()

    main(args)

