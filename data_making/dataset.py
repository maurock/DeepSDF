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
    TODO: adapting to handle all objects
    """
    def __init__(self):
        samples_dict = np.load(os.path.join(os.path.dirname(results.__file__), 'samples_dict.npy'), allow_pickle=True).item()
        for obj_idx in list(samples_dict.keys())[:1]:  # samples_dict.keys() for all the objects
            self.data = dict()
            for key in samples_dict[obj_idx].keys():   # keys are 'samples', 'sdf'
                self.data[key] = torch.from_numpy(samples_dict[obj_idx][key]).to(device)
        return

    def __len__(self):
        return self.data['samples'].shape[0]

    def __getitem__(self, idx):
        verts = self.data['samples'][idx, :]
        sdf = self.data['sdf'][idx]
        return verts, sdf

if __name__=='__main__':
    dataset = SDFDataset()
