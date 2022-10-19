import torch.nn as nn
import torch
"""
Model based on the paper 'DeepSDF'
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SDFModel(torch.nn.Module):
    def __init__(self):
        super(SDFModel, self).__init__()
        # MLP
        layers = []
        torch.utils.weight_norm(nn.Linear(20, 40))
        layers.append(nn.Sequential(torch.utils.weight_norm(nn.Linear(3, 512)), nn.ReLU()))
        layers.append(nn.Sequential(torch.utils.weight_norm(nn.Linear(512, 512)), nn.ReLU()))
        layers.append(nn.Sequential(torch.utils.weight_norm(nn.Linear(512, 512)), nn.ReLU()))
        layers.append(nn.Sequential(torch.utils.weight_norm(nn.Linear(512, 512)), nn.ReLU()))
        layers.append(nn.Sequential(torch.utils.weight_norm(nn.Linear(512, 512)), nn.ReLU()))
        layers.append(nn.Sequential(torch.utils.weight_norm(nn.Linear(512, 512)), nn.ReLU()))
        layers.append(nn.Sequential(torch.utils.weight_norm(nn.Linear(512, 512)), nn.ReLU()))
        layers.append(nn.Sequential(torch.utils.weight_norm(nn.Linear(512, 1)), nn.Tanh()))
        self.fc = nn.Sequential(*layers)

    def forward(self, points):
        sdf = self.fc(points)
        return sdf