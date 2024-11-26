import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from model.PTv3 import Point
from model.LidarUpsample import Lidar4US, Point


def load_kitti_bin(file_path):
    """
    Load a KITTI .bin file as a numpy array.
    :param file_path: Path to .bin file
    :return: Numpy array of shape (N, 4) [x, y, z, intensity]
    """
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def kitti_to_dict(file_path, grid_size=0.05, batch_id=0):
    """
    Convert a KITTI .bin file to a Point object suitable for the model.
    :param file_path: Path to .bin file
    :param grid_size: Grid size for quantization
    :param batch_id: Batch ID for the point cloud
    :return: Point object
    """
    raw_data = load_kitti_bin(file_path)
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity as a feature

    # Combine coords and intensity as features
    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(intensity, dtype=torch.float32)
    ], dim=1)

    batch_tensor = torch.full((features.shape[0],), batch_id, dtype=torch.int64)

    # Create Point object
    data_dict = {
        "coord": features[:, :3].to("cuda"),  # Coordinates (x, y, z)
        "feat": features[:,:3].to("cuda"),
        "batch": batch_tensor.to("cuda"),  # Batch IDs
        "grid_size" : torch.tensor(0.01).to("cuda")
    }
    return data_dict

def kitti_to_dict(file_path, grid_size=0.05, batch_id=0):
    raw_data = load_kitti_bin(file_path)
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity as a feature

    # Combine coords and intensity as features
    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(intensity, dtype=torch.float32)
    ], dim=1)

    batch_tensor = torch.full((features.shape[0],), batch_id, dtype=torch.int64)

    # Create Point object
    data_dict = {
        "coord": features[:, :3].to("cuda"),  # Coordinates (x, y, z)
        "feat": features.to("cuda"),  # Coordinates (x, y, z) + intensity
        "batch": batch_tensor.to("cuda"),  # Batch IDs
        "grid_size" : torch.tensor(0.01).to("cuda")
    }
    return data_dict


class PointCloudDataset(Dataset):
    def __init__(self, file_paths, grid_size=0.05):
        self.file_paths = file_paths
        self.grid_size = grid_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        batch_id = idx
        return kitti_to_dict(file_path, grid_size=self.grid_size, batch_id=batch_id)


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_points, gt_points):
        max_pcl_len = min(len(pred_points.feat,gt_points["feat"]))

        pred_p = pred_points.feat[:max_pcl_len]
        gt_p = gt_points["coord"][:max_pcl_len]

        

        return torch.mean(loss)







## !!gt and train has the same filename!!
train_file_paths = glob("/home/server01/js_ws/dataset/sparse_pointclouds_kitti/train/*.bin", recursive=True)
gt_file_paths = glob("/home/server01/js_ws/dataset/sparse_pointclouds_kitti/GT/*.bin")

train_dataset = PointCloudDataset(train_file_paths)
gt_dataset = PointCloudDataset(gt_file_paths)

model = Lidar4US(
    in_channels=4,  #coord + intensity
    drop_path=0.3,
    block_depth=(2, 2, 2, 6, 6, 2),
    enc_channels=(32, 64, 128, 64, 32, 16),
    enc_n_heads=(2, 4, 8, 16, 16, 8),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024, 1024),
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    mlp_ratio=4,
    stride=(2, 2, 4, 4, 4),
    dec_depths=(2, 2, 2, 2, 2),
    dec_n_head=(4, 4, 8, 16, 32),
    dec_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_channels=(128, 128, 256, 256, 512),
    train_decoder=True,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    upsample_ratio=32,
    out_channel = 3,
)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

num_epochs = 10
device = torch.device("cuda")

model.to(device)
# loss_fn.to(device)

## TEST 1 just make x30 size output(11/22)

# serialized_gt = []
orders=("z", "z-trans", "hilbert", "hilbert-trans")

# for gt in gt_dataset:
#     p_gt = Point(gt)
#     tmp = []
#     p_gt.serialization(orders,shuffle_orders=False)
#     serialized_gt.append(p_gt)
# p_gt = Point(gt_dataset[0])
# p_gt.serialization(orders,shuffle_orders=False)

# print("raw: ")
# print(len(gt_dataset[0]["feat"]))
# print("code :")
# print(p_gt.serialized_code[0])
# print("order :")
# print(max(p_gt.serialized_order[0]))


# print(torch.concat(train_dataset[0]["feat"][0],torch.tensor([0,0,0,0]),dim = 1))
# sys.exit()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i in range(len(train_dataset)):
        pred = model(train_dataset[i])
        print(pred.feat.shape)
        print(gt_dataset[i]["coord"].shape)


    sys.exit()



    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")