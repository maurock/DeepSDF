import torch.nn as nn
import torch
"""
Model based on the paper 'DeepSDF'
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SDFModel(torch.nn.Module):
    def __init__(self, input_dim=3, inner_dim=512, output_dim=1):
        super(SDFModel, self).__init__()
        # MLP
        layers = []
        for _ in range(7):
            layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), nn.ReLU()))
            input_dim = inner_dim
        layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, output_dim)), nn.Tanh()))
        self.fc = nn.Sequential(*layers)

    def forward(self, points):
        sdf = self.fc(points)
        return sdf