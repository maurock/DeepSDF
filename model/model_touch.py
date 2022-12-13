import torch.nn as nn
import torch
"""
Models adapted from 'Active 3D Shape Reconstruction from Vision and Touch, Smith et al.'
https://arxiv.org/abs/2107.09584
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# adapted from https://github.com/facebookresearch/Active-3D-Vision-and-Touch/blob/main/pterotactyl/reconstruction/touch/model.py
# CNN block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()
        self.last = last
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
        )
        self.activation = nn.Sequential(
             nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        if not self.last:
            x = self.activation(x)
        return x

# adapted from https://github.com/facebookresearch/Active-3D-Vision-and-Touch/blob/main/pterotactyl/reconstruction/touch/model.py
# Model for predicting touch chart shape
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # CNN
        CNN_layers = []
        CNN_layers.append(DoubleConv(1, 32))                # in: 256, out:64
        CNN_layers.append(DoubleConv(32, 16))               # in: 64, out:16
        CNN_layers.append(DoubleConv(16, 32, last=True))    # in:16, out:4
        self.CNN_layers = nn.Sequential(*CNN_layers)

        # MLP
        layers = []
        layers.append(nn.Sequential(nn.Linear(512, 256), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(256, 128), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(128, 75)))
        self.fc = nn.Sequential(*layers)

    def predict_verts(self, touch):
        for layer in self.CNN_layers:
            touch = layer(touch)
        points = touch.contiguous().view(-1, 512)
        points = self.fc(points)
        return points

    def forward(self, tactile_img, initial_verts):
        verts = initial_verts + self.predict_verts(tactile_img).view(-1, initial_verts.shape[1], 3)
        return verts