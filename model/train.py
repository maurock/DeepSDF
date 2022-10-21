import torch.nn as nn
import torch
import model.sdf_model as sdf_model
import torch.optim as optim
import data_making.dataset as dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import argparse
import results.runs as runs
from utils.utils import SDFLoss
import os
from datetime import datetime
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, args):
        self.args = args

    def __call__(self):
        # directories
        self.timestamp_run = datetime.now().strftime('%d_%m_%H%M')   # timestamp to use for logging data
        self.runs_dir = os.path.dirname(runs.__file__)               # directory fo all runs
        self.run_dir = os.path.join(self.runs_dir, self.timestamp_run)  # directory for this run
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        self.model = sdf_model.SDFModel().double().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0)
        # get data
        train_loader, val_loader = self.get_loaders()
        self.results = dict()
        self.results['train'] = []
        self.results['val'] = []
        for epoch in range(self.args.epochs):
            print(f'============================ Epoch {epoch} ============================')
            self.epoch = epoch
            avg_train_loss = self.train(train_loader)
            self.results['train'].append(avg_train_loss)
            with torch.no_grad():
                avg_val_loss = self.validate(val_loader)
                self.results['val'].append(avg_val_loss)
            np.save(os.path.join(self.run_dir, 'results.npy'), self.results)
            torch.save(self.model.state_dict(), os.path.join(self.run_dir, 'weights.pt'))
    
    def get_loaders(self):
        data = dataset.SDFDataset()
        train_size = int(0.8 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = random_split(data, [train_size, val_size])
        train_loader = DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=True
            )
        val_loader = DataLoader(
            val_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=True
            )
        return train_loader, val_loader
    
    def train(self, train_loader):
        total_loss = 0.0
        iterations = 0.0
        self.model.train()
        loss = nn.MSELoss()
        for batch in train_loader:
            iterations += 1.0
            #batch_size = self.args.batch_size
            self.optimizer.zero_grad()
            x = batch[0]                 # (batch_size, 3)
            y = batch[1].view(-1, 1)     # (batch_size, 1)
            predictions = self.model(x)  # (batch_size, 1)
            loss_value = loss(y, predictions)
            loss_value.backward()
            self.optimizer.step()
            total_loss += loss_value.data.cpu().numpy()      
        avg_train_loss = total_loss/iterations
        print(f'Training: loss {avg_train_loss}')
        return avg_train_loss

    def validate(self, val_loader):
        total_loss = 0.0
        iterations = 0.0
        self.model.eval()
        for batch in val_loader:
            iterations+=1.0
            #batch_size = self.args.batch_size
            self.optimizer.zero_grad()
            x = batch[0]
            y = batch[1].view(-1, 1)     
            predictions = self.model(x)
            loss_value = SDFLoss(y, predictions)
            total_loss += loss_value.data.cpu().numpy()      
        avg_val_loss = total_loss/iterations
        print(f'Validation: loss {avg_val_loss}')
        return avg_val_loss

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="Setting for the random seed."
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to use."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Size of the batch."
    )
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer()