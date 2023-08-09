import numpy as np
import torch
from torch.utils.data import Dataset
import os
import results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SDFDataset(Dataset):
    """
    TODO: adapting to handle multiple objects
    """
    def __init__(self, dataset_name, limit_data=1):
        samples_dict = np.load(os.path.join(os.path.dirname(results.__file__), f'samples_dict_{dataset_name}.npy'), allow_pickle=True).item()

        obs_idxs = list(samples_dict.keys())
        # Limit the number of objects to use for training
        if limit_data<1:
            obs_idxs = obs_idxs[:int(limit_data*len(obs_idxs))]

        self.data = dict()
        for obj_idx in obs_idxs:  # samples_dict.keys() for all the objects
            for key in samples_dict[obj_idx].keys():   # keys are ['samples', 'sdf', 'latent_class', 'samples_latent_class']
                value = torch.from_numpy(samples_dict[obj_idx][key]).float().to(device)
                if len(value.shape) == 1:    # increase dim if monodimensional, needed to vstack
                    value = value.view(-1, 1)
                if key not in list(self.data.keys()):
                    self.data[key] = value
                else:
                    self.data[key] = torch.vstack((self.data[key], value))
        return

    def __len__(self):
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx):
        latent_class = self.data['samples_latent_class'][idx, :]
        sdf = self.data['sdf'][idx]
        return latent_class, sdf

if __name__=='__main__':
    dataset_name = "ShapeNetCore"
    dataset = SDFDataset(dataset_name)
