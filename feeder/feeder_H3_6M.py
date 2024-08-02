import torch
import numpy as np
import glob
import os
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from tools import Augmenter3D
from tools import read_pkl
from tools import flip_data
    
class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]  
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
                if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')    
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), index


if __name__ == '__main__':
    import argparse
    args = argparse.Namespace()
    args.flip = True
    args.synthetic = True
    args.aug = Augmenter3D(args)
    args.gt_2d = False
    args.data_root = '/media/os/data/ruihang'
    args.subset_list = ['H36M-SH']

    db = MotionDataset3D(args=args, subset_list=args.subset_list, data_split='train')
    batch_input, batch_gt, index = next(iter(db))
    print('sample: ', batch_input.shape, batch_gt.shape, index)
    data_numpy, label, index = db.__getitem__(51)
    print('sample: ', batch_input.shape, batch_gt.shape, index)
    data_numpy, label, index = db.__getitem__(52)
    print('sample: ', batch_input.shape, batch_gt.shape, index)
    print(batch_input[:, :, 0])
    print(batch_input[:, :, 1])
    print(batch_input[:, :, 2])

    db = MotionDataset3D(args=args, subset_list=args.subset_list, data_split='test')
    batch_input, batch_gt, index = next(iter(db))
    print('sample: ', batch_input.shape, batch_gt.shape, index)
    data_numpy, label, index = db.__getitem__(51)
    print('sample: ', batch_input.shape, batch_gt.shape, index)
    data_numpy, label, index = db.__getitem__(52)
    print('sample: ', batch_input.shape, batch_gt.shape, index)