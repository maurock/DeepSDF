import torch.nn as nn
import torch
import copy
from tqdm import tqdm
from utils import utils_deepsdf
import numpy as np
"""
Model based on the paper 'DeepSDF'
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SDFModel(torch.nn.Module):
    def __init__(self, input_dim=3, inner_dim=512, output_dim=1):
        """
        SDF model for a signle shape.
        Args:
            input_dim: 3 dimensions (point coordinate)
        """
        super(SDFModel, self).__init__()
        # MLP
        layers = []
        for _ in range(7):
            layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), nn.ReLU()))
            input_dim = inner_dim
        layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, output_dim)), nn.Tanh()))
        self.net = nn.Sequential(*layers)

    def forward(self, points):
        sdf = self.net(points)
        return sdf


class SDFModelMulti(torch.nn.Module):
    def __init__(self, num_layers, no_skip_connections, positional_encoding_embeddings, latent_size, inner_dim=512, output_dim=1):
        """
        SDF model for multiple shapes.
        Args:
            input_dim: 128 for latent space + 3 points = 131
        """
        super(SDFModelMulti, self).__init__()

        # Num layers of the entire network
        self.num_layers = num_layers 
        # If skip connections, add the input to one of the inner layers
        self.skip_connections = not no_skip_connections
        # Dimension of the input space: when using positional encoding, the input size is 3 + 6 * positional_encoding_embeddings
        dim_coords = 3 if positional_encoding_embeddings == 0 else 3 + 6 * positional_encoding_embeddings
        input_dim = latent_size + dim_coords
        # Copy input size to calculate the skip tensor size
        self.skip_tensor_dim = copy.copy(input_dim)
        # Compute how many layers are not Sequential
        num_extra_layers = 2 if (self.skip_connections and self.num_layers >= 8) else 1
        # Add sequential layers
        layers = []
        for _ in range(num_layers - num_extra_layers):
            layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), nn.ReLU()))
            input_dim = inner_dim
        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(nn.Linear(inner_dim, output_dim), nn.Tanh())
        self.skip_layer = nn.Sequential(nn.Linear(inner_dim, inner_dim - self.skip_tensor_dim), nn.ReLU())
        self.positional_encoding_embeddings = positional_encoding_embeddings

    # Apply positional encoding to increase the dimensionality of the input
    def positional_encoding(self, points):
        embeddings = []
        embeddings.append(torch.sin(np.pi * points))
        embeddings.append(torch.cos(np.pi * points))
        for i in range(1, self.positional_encoding_embeddings):
            embeddings.append(torch.sin(np.pi * 2 * i * points))
            embeddings.append(torch.cos(np.pi * 2 * i * points))
        embeddings.append(points)
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings

    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, 131). It contains a stacked tensor [latent_code, samples].
        Returns:
            sdf: output tensor of shape (batch_size, 1)
        """
        # Add positional encoding
        if self.positional_encoding_embeddings > 0:
            # x[:, -3:] contains the samples
            embeddings = self.positional_encoding(x[:, -3:]).to(device)
            x = torch.cat((x[:, :-3], embeddings), dim=-1)
        
        input_data = x.clone().detach()

        # Forward pass
        if self.skip_connections and self.num_layers >= 8:
            for i in range(3):
                x = self.net[i](x)
            x = self.skip_layer(x)
            x = torch.hstack((x, input_data))
            for i in range(self.num_layers - 5):
                x = self.net[3 + i](x)
            sdf = self.final_layer(x)
        else:
            if self.skip_connections:
                print('The network requires at least 8 layers to skip connections. Normal forward pass is used.')
            x = self.net(x)
            sdf = self.final_layer(x)
        return sdf

    
    def initialise_latent_code(self, latent_size):
        """Initialise latent code with random noise."""
        latent_code = torch.normal(0, 0.01, size = (1, latent_size), dtype=torch.float32, requires_grad=True, device=device)

        return latent_code


    def infer_latent_code(self, args, coords, sdf_gt, writer, latent_code):
        """Infer latent code from coordinates, their sdf, and a trained model."""

        # Initialise latent code and optimiser
        #latent_code = self.initialise_latent_code(args.latent_size)
        
        if args.optimiser == 'Adam':
            optim = torch.optim.Adam([latent_code], lr=args.lr)
        elif args.optimiser == 'LBFGS':
            optim = torch.optim.LBFGS([latent_code], lr=args.lr, max_iter=args.LBFGS_maxiter)
        else:
            print('Please choose valid optimiser: [Adam, LBFGS]')
            exit()

        if args.lr_scheduler:
            scheduler_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', 
                                                    factor=args.lr_multiplier, 
                                                    patience=args.patience, 
                                                    threshold=0.001, threshold_mode='rel')

        best_loss = 1000000

        for epoch in tqdm(range(0, args.epochs)):

            latent_code_tile = torch.tile(latent_code, (coords.shape[0], 1))
            x = torch.hstack((latent_code_tile, coords))

            # Adam 
            if args.optimiser == 'Adam':

                optim.zero_grad()

                predictions = self(x)

                if args.clamp:
                    predictions = torch.clamp(predictions, -args.clamp_value, args.clamp_value)

                loss_value, l1, l2 = utils_deepsdf.SDFLoss_multishape(sdf_gt, predictions, x[:, :args.latent_size], sigma=args.sigma_regulariser)
                loss_value.backward()

                writer.add_scalar('Reconstruction loss', l1.data.cpu().numpy(), epoch)
                writer.add_scalar('Latent code loss', l2.data.cpu().numpy(), epoch)

                
                #  add langevin noise (optional)
                if args.langevin_noise > 0:
                    noise = torch.normal(0, args.langevin_noise, size = (1, args.latent_size), dtype=torch.float32, requires_grad=False, device=device)
                    latent_code.grad = latent_code.grad + noise

                optim.step()

            # LBFGS
            else:

                def closure():
                    optim.zero_grad()

                    predictions = self(x)

                    if args.clamp:
                        predictions = torch.clamp(predictions, -args.clamp_value, args.clamp_value)

                    loss_value, l1, l2 = utils_deepsdf.SDFLoss_multishape(sdf_gt, predictions, x[:, :args.latent_size], sigma=args.sigma_regulariser)
                    loss_value.backward()

                    return loss_value

                optim.step(closure)

                loss_value = closure()

            if l1.detach().cpu().item() < best_loss:
                best_loss = l1.detach().cpu().item()
                best_latent_code = latent_code.clone()

            # step scheduler and store on tensorboard (optional)
            if args.lr_scheduler:
                scheduler_latent.step(loss_value.item())
                writer.add_scalar('Learning rate', scheduler_latent._last_lr[0], epoch)

                if scheduler_latent._last_lr[0] < 1e-6:
                    print('Learning rate too small, stopping training')
                    break

            # logging
            writer.add_scalar('Training loss', loss_value.detach().cpu().item(), epoch)
            # store latent codes and their gradient on tensorboard
            #tag = f"latent_code_0"
            #writer.add_histogram(tag, latent_code, global_step=epoch)
            #tag = f"grad_latent_code_0"
            #writer.add_histogram(tag, latent_code.grad, global_step=epoch)

        return best_latent_code