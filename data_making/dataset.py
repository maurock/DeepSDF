import numpy as np
import torch
from torch.utils.data import Dataset
import os
import results
from glob import glob
from model.sdf_model import SDFModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SDFDataset(Dataset):
    """
    TODO: adapting to handle multiple objects
    """
    def __init__(self):
        samples_dict = np.load(os.path.join(os.path.dirname(results.__file__), 'samples_dict.npy'), allow_pickle=True).item()
        self.data = dict()
        for obj_idx in list(samples_dict.keys()):  # samples_dict.keys() for all the objects
            for key in samples_dict[obj_idx].keys():   # keys are ['samples', 'sdf', 'latent_class', 'samples_latent_class']
                value = samples_dict[obj_idx][key]
                # convert value to np.array if not already
                self.data[key] = torch.from_numpy(value).to(device)
        return

    def __len__(self):
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx):
        latent_class = self.data['samples_latent_class'][idx, :]
        sdf = self.data['sdf'][idx]
        return latent_class, sdf

if __name__=='__main__':
    dataset = SDFDataset()
