import numpy as np
from torch.utils.data import Dataset
import torch
import random
import feeder.tools as tools
# import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=[1], split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, random_spatial_flip=False, window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=True, vel=False):
        """
        data_path:
        label_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_spatial_flip = random_spatial_flip
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        self.index = 0
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)        # (N, 3, T, 25, 2)


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)
        

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        p = 0.5
        if self.random_move:
            if random.random() < p:
                data_numpy = tools.random_move(data_numpy)

        if self.random_spatial_flip:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if self.random_rot:
            if random.random() < p:
                data_numpy = tools.random_rot(data_numpy)

        if self.bone:
            ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        # (N, 3, T, 25, 2) -> (N, 2, T, 25, 3)
        if isinstance(data_numpy, np.ndarray):
            data_numpy = torch.from_numpy(data_numpy.copy())
        elif isinstance(data_numpy, torch.Tensor):
            data_numpy = data_numpy.clone()
        else:
            raise TypeError("Unsupported data type. Only NumPy arrays or PyTorch tensors are supported.")
        # return torch.from_numpy(np.transpose(data_numpy, (3, 1, 2, 0))).float().clone().detach(), label, index
        return data_numpy.float().clone().detach(), label, index
        



if __name__ == '__main__':

    # db = Feeder(data_path='/home/ruihang/Skeleton/Code/Refactor/data/NTU60_XSub.npz', split='train', window_size=100,
    #         random_shift=False, random_move=True, random_spatial_flip=True, random_rot=True)
    db = Feeder(data_path='/home/ruihang/Skeleton/Code/Codes/dataset/NTU60/NTU60_XSub_kf.npz', 
                    split='train', window_size=120,
                    random_shift=False, random_move=True, random_spatial_flip=True, random_rot=True)
    # db2 = Feeder(data_path='/home/ruihang/Skeleton/Code/Codes/dataset/NTU60/NTU60_XSub_kf2.npz', 
    #                 split='train', window_size=120,
    #                 random_shift=False, random_move=True, random_spatial_flip=True, random_rot=True)
    data_numpy, label, index = next(iter(db))
    print('sample: ', data_numpy.shape, label, index)
    data_numpy, label, index = db.__getitem__(104)
    print('sample: ', data_numpy.shape, label, index)
    data_numpy, label, index = db.__getitem__(105)
    print('sample: ', data_numpy.shape, label, index)
    data_numpy, label, index = db.__getitem__(106)
    print('sample: ', data_numpy.shape, label, index)
    data_numpy, label, index = db.__getitem__(107)
    print('sample: ', data_numpy.shape, label, index)