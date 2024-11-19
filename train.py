from glob import glob
from inputprocess.lidar_to_tensor import KiTTILoader
from torch.utils.data import DataLoader

file_paths = glob("/home/junseo/datasets/kitti/**/*.bin", recursive=True)
dataset = KiTTILoader(file_paths)
print(dataset)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    batch_as_list = [points for points in batch]
    for points in batch_as_list:
        print(points.shape)