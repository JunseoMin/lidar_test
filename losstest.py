import torch
import torch.nn as nn

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


# def kitti_to_dict(file_path, grid_size=0.05, batch_id=0):
    # """
    # Convert a KITTI .bin file to a Point object suitable for the model.
    # :param file_path: Path to .bin file
    # :param grid_size: Grid size for quantization
    # :param batch_id: Batch ID for the point cloud
    # :return: Point object
    # """
    # raw_data = load_kitti_bin(file_path)
    # coords = raw_data[:, :3]  # x, y, z
    # intensity = raw_data[:, 3:4]  # intensity as a feature

    # # Combine coords and intensity as features
    # features = torch.cat([
    #     torch.tensor(coords, dtype=torch.float32),
    #     torch.tensor(intensity, dtype=torch.float32)
    # ], dim=1)

    # batch_tensor = torch.full((features.shape[0],), batch_id, dtype=torch.int64)

    # # Create Point object
    # data_dict = {
    #     "coord": features[:, :3].to("cuda"),  # Coordinates (x, y, z)
    #     "feat": features[:,:3].to("cuda"),
    #     "batch": batch_tensor.to("cuda"),  # Batch IDs
    #     "grid_size" : torch.tensor(0.01).to("cuda")
    # }
    # return data_dict

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

gt_path="/home/server01/js_ws/dataset/sparse_pointclouds_kitti/GT/0000000080_14298.bin"


# 임의의 예측값 생성
pred = kitti_to_dict(gt_path)

# Ground Truth를 예측값과 동일하게 설정
gt = pred["coord"].clone().detach()

class LidarUpsampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_points, gt_points):
        max_len = min(len(pred_points["coord"]), len(gt_points))
        pred = pred_points["coord"][:max_len]
        gt = gt_points[:max_len]
        loss = self.criterion(pred, gt)
        return loss

class LidarUpsampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_points, gt_points):
        max_len = min(len(pred_points["feat"]), len(gt_points["feat"]))
        print(max_len)
        print(pred_points.shape)
        print(gt_points.shape)
        pred = pred_points["feat"][:max_len]
        print(pred.shape)
        gt = gt_points["feat"][:max_len]
        print(gt.shape)
        loss = self.criterion(pred, gt)
        return loss

# MSELoss 정의
criterion = LidarUpsampleLoss()

# Loss 계산
print(pred["coord"].shape)
print(gt.shape)


# loss = criterion(pred["coord"], gt)

# print(f"Loss with identical pred and GT: {loss.item():.6f}")

# Backpropagation 테스트
# loss.backward()
# print(f"Gradients on predictions (should not be zero): {pred.grad}")
