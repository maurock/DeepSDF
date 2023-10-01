import numpy as np
import torch
from torch.utils.data import Dataset
import results
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TouchChartDataset(Dataset):
    def __init__(self, touch_charts_path):
        """
        Check self.get_total_data() for details on touch_charts_dir and the required file tree.
        """
        self.data = np.load(touch_charts_path, allow_pickle=True)
        for key in ['pointclouds', 'tactile_imgs']:
            self.data[key] = torch.tensor(self.data[key], dtype=torch.float32, device=device)
        return

    def __len__(self):
        return self.data['pointclouds'].shape[0]

    def __getitem__(self, idx):
        x = self.data['tactile_imgs'][idx, :, :]
        y = self.data['pointclouds'][idx, :, :]
        return x, y

    def _analyse_dataset(self):
        """
        Print information about full dataset
        """
        print(f"The dataset has shape {self.data['tactile_imgs'].shape}")
        print(f"The ground truth pointcloud for one instance has a shape {self.data['pointclouds'][0].shape}")
    

if __name__=='__main__':
    touch_charts_path = os.path.join(os.path.dirname(results.__file__), 'touch_charts_gt.npy')
    dataset = TouchChartDataset(touch_charts_path)
    print(dataset[0])
