import numpy as np

data_path = '/home/ruihang/Skeleton/Code/Codes/dataset/NTU60/NTU60_XSub_kf.npz'
npz_data = np.load(data_path, mmap_mode='r')
data_path = '/home/ruihang/Skeleton/Code/Codes/dataset/NTU60/NTU60_XSub_kf.npz'
npz_data_kf = np.load(data_path, mmap_mode='r')

print((npz_data['x_train']-npz_data_kf['x_train']).sum())
print((npz_data['x_test']-npz_data_kf['x_test']).sum())