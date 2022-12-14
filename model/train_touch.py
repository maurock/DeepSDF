"""
Train local touch chart model. 
"""
import numpy as np
import os
from torch.utils.data import DataLoader
from data_making.dataset_touch import TouchChartDataset
import data
import argparse
import results
from torch.utils.data import random_split
from model import model_touch
import torch 
import torch.optim as optim
from pytorch3d.io.obj_io import load_obj
from datetime import datetime
import json
from sklearn.model_selection import KFold
from utils import utils_misc, utils_mesh 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, args):
        utils_misc.set_seeds(41)
        self.touch_chart_path = os.path.join(os.path.dirname(results.__file__), 'touch_charts_gt.npy')
        self.args = args
        # load initial mesh sheet to deform using the Encoder
        chart_location = os.path.join(os.path.dirname(data.__file__), 'touch_chart.obj')
        self.initial_verts, self.initial_faces = utils_mesh.load_mesh_touch(chart_location)
        # repeat as many times as the batch size. This is because we need to add the initial_verts to the prediction for every element in the batch.        
        self.initial_verts = self.initial_verts.view(1, self.initial_verts.shape[0], 3).repeat(
            args.batch_size, 1, 1
        )
        # Path debug
        self.timestamp_run = datetime.now().strftime('%d_%m_%H%M')   # timestamp to use for logging data
        self.debug_val_path = os.path.join(os.path.dirname(results.__file__), 'touch_val_debug', f'val_dict_{self.timestamp_run}')
        self.results = dict()
        self.fold = 0

    def __call__(self):
        if self.args.log_info_train:         # log info train
            # Create folder to store files
            self.log_train_dir = os.path.join(os.path.dirname(results.__file__), "runs_touch", f"{self.timestamp_run}")
            if not os.path.exists(self.log_train_dir):
                os.mkdir(self.log_train_dir)
            # Create log. This will be populated with settings, losses, etc..
            self.log_path = os.path.join(self.log_train_dir, "settings.txt")
            args_dict = vars(self.args)  # convert args to dict to write them as json
            with open(self.log_path, mode='a') as log:
                log.write('Settings:\n')
                log.write(json.dumps(args_dict).replace(', ', ',\n'))
                log.write('\n\n')        
        full_dataset = TouchChartDataset(self.touch_chart_path)
        self.encoder = model_touch.Encoder().to(device)

        # Cross validation
        if self.args.cross_validation:
            kfold = KFold(n_splits=self.args.k_folds, shuffle=True)
            for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
                # Set variables to store results
                self.fold = fold
                self.results[self.fold] = dict()
                self.results[self.fold]['train'] = []
                self.results[self.fold]['val'] = []
                print(f'Fold: {self.fold}')
                train_loader, val_loader = self.get_loaders_cv(full_dataset, train_ids, val_ids)
                for epoch in range(self.args.epochs):
                    print(f'============================ Epoch {epoch} ============================')
                    self.epoch = epoch
                    self.train(train_loader)
                    self.validate(val_loader)
        # Single training
        else:
            self.results[0] = dict()
            self.results[0]['train'] = []
            self.results[0]['val'] = []
            train_loader, val_loader = self.get_loaders(full_dataset)
            for epoch in range(self.args.epochs):
                print(f'============================ Epoch {epoch} ============================')
                self.epoch = epoch
                self.train(train_loader)
                with torch.no_grad():
                    self.validate(val_loader)

    def get_loaders_cv(self, full_dataset, train_ids, val_ids):
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        if self.args.analyse_dataset:
            full_dataset._analyse_dataset()    # print info about dataset
        train_loader = DataLoader(
                full_dataset,
                batch_size=self.args.batch_size,
                num_workers=1,
                sampler=train_subsampler
            )
        val_loader = DataLoader(
            full_dataset,
            batch_size=self.args.batch_size,
            num_workers=1,
            sampler=val_subsampler,
            drop_last=True
            )
        return train_loader, val_loader
    
    def get_loaders(self, full_dataset):
        if self.args.analyse_dataset:
            full_dataset._analyse_dataset()    # print info about dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_data, val_data = random_split(full_dataset, [train_size, val_size])
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
        total_loss = 0
        iterations = 0
        self.encoder.train()
        params = list(self.encoder.parameters())

        self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=0)

        for batch in train_loader:
            iterations += 1
            batch_size = batch[0].shape[0]
            # batch is a list containing X and Y
            self.optimizer.zero_grad()
            tactile_imgs = batch[0]
            pointcloud_gt = batch[1]    # this is [batch_size, N points, 3]
            pred_verts = self.encoder(tactile_imgs, self.initial_verts.clone()[:batch_size])

            loss = self.args.loss_coeff * utils_mesh.chamfer_distance(
                pred_verts, self.initial_faces, pointcloud_gt, self.args.num_samples
            )            
            loss = loss.mean()
            # backprop
            loss.backward()
            # clip gradient to avoid grad explosion
            clip_value=0.05
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_value)
            # step
            self.optimizer.step()

            total_loss += loss.data.cpu().numpy()      

        print(f'Training: loss {total_loss/iterations}')

        self.results[self.fold]['train'].append(total_loss/iterations)
        np.save(os.path.join(self.log_train_dir, 'results_dict.npy'), self.results)
        if self.args.log_info_train:
            with open(self.log_path, mode='a') as log:
                log.write(f'Fold {self.fold}, Epoch {self.epoch}, Train loss: {total_loss / iterations} \n')
            torch.save(self.encoder.state_dict(), os.path.join(self.log_train_dir, 'weights.pt'))

    def validate(self, valid_loader):
        total_loss = 0
        self.encoder.eval()
        iterations = 0

        # Lists for debug
        pred_points_list = np.array([]).reshape(0, 1000, 3)
        pred_verts_list = np.array([]).reshape(0, 25, 3)
        pointcloud_gt_list = np.array([]).reshape(0, 2000, 3)

        for batch in valid_loader:
            iterations += 1
            batch_size = batch[0].shape[0]
            # batch is a list containing X and Y
            tactile_imgs = batch[0]
            pointcloud_gt = batch[1]
            pred_verts = self.encoder(tactile_imgs, self.initial_verts.clone()[:batch_size])
            loss = self.args.loss_coeff * utils_mesh.chamfer_distance(
                pred_verts, self.initial_faces, pointcloud_gt, self.args.num_samples
            )
            loss = loss.mean()
            total_loss += loss.data.cpu().numpy()

            if self.args.debug_validation:
                pred_points= utils_mesh.batch_sample(pred_verts, self.initial_faces, num=1000)
                pred_points_list = np.vstack((pred_points_list, pred_points.cpu().numpy()))
                pred_verts_list = np.vstack((pred_verts_list, pred_verts.cpu().numpy()))
                pointcloud_gt_list = np.vstack((pointcloud_gt_list, pointcloud_gt.cpu().numpy()))
                val_dict = dict()
                val_dict['pred_points'] = pred_points_list
                val_dict['pred_verts'] = pred_verts_list
                val_dict['pointcloud_gt'] = pointcloud_gt_list
                np.save(self.debug_val_path, val_dict)

        print(f'Validation: loss {total_loss/iterations}')
        self.results[self.fold]['val'].append(total_loss/iterations)
        np.save(os.path.join(self.log_train_dir, 'results_dict.npy'), self.results)
        if self.args.log_info_train:
            with open(self.log_path, mode='a') as log:
                log.write(f'Epoch {self.epoch}, Val loss: {total_loss / iterations} \n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="Setting for the random seed."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to use."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Size of the batch."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of points in the predicted point cloud.",
    )
    parser.add_argument(
        "--loss_coeff", type=float, default=1.0, help="Coefficient for loss term."
    )
    parser.add_argument(
        "--analyse_dataset",
        action='store_true',
        default=False,
        help="print info about training dataset",
    )
    parser.add_argument(
        "--debug_validation",
        action='store_true',
        default=False,
        help="Store data to visualise the results of the model on the validation set",
    )
    parser.add_argument(
        "--log_info_train",
        action='store_true',
        default=False,
        help="Log info about training",
    ) 
    parser.add_argument(
        "--cross_validation",
        action='store_true',
        default=False,
        help="Set cross validation to True or False",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=3,
        help="Number of folds",
    )
    args = parser.parse_args()

    args.batch_size = 5
    args.analyse_dataset = True
    args.log_info_train = True

    trainer = Trainer(args)
    trainer()