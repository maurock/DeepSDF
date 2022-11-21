import torch.nn as nn
import torch
import copy
"""
Model based on the paper 'DeepSDF'
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, num_layers, no_skip_connections, input_dim=131, inner_dim=512, output_dim=1):
        """
        SDF model for multiple shapes.
        Args:
            input_dim: 128 for latent space + 3 points = 131
        """
        super(SDFModelMulti, self).__init__()
        # MLP
        # layers = []
        # if skip_connections and num_layers >= 8:
        #     layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), nn.ReLU()))
        #     for _ in range(3):
        #         layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(inner_dim, inner_dim)), nn.ReLU()))
        #     layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(inner_dim, inner_dim - input_dim)), nn.ReLU()))
        #     for _ in range(3):
        #         layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(inner_dim, inner_dim)), nn.ReLU()))
        # else:
        #     if skip_connections:
        #         print('The model requires at least 8 layers to skip connections. Build the network without skipping connections.')
        #     for _ in range(num_layers-1):
        #         #layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), nn.ReLU()))
        #         layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), nn.ReLU()))
        #         input_dim = inner_dim
        # #layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, output_dim)), nn.Tanh()))
        # layers.append(nn.Sequential(nn.Linear(input_dim, output_dim), nn.Tanh()))


        self.num_layers = num_layers
        self.skip_connections = not no_skip_connections
        self.skip_tensor_dim = copy.copy(input_dim)
        num_extra_layers = 2 if (self.skip_connections and self.num_layers >= 8) else 1
        layers = []
        for _ in range(num_layers - num_extra_layers):
            layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), nn.ReLU()))
            input_dim = inner_dim
        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(nn.Linear(inner_dim, output_dim), nn.Tanh())
        self.skip_layer = nn.Sequential(nn.Linear(inner_dim, inner_dim - self.skip_tensor_dim), nn.ReLU())

    def forward(self, x):
        input_data = x.clone().detach()
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
