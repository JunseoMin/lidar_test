import torch
import numpy as np
from torch.utils.data import Dataset

def load_kitti_bin(file_path):
    # print(file_path)
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def load_kitti_bin_gt(file_path):
    # print(file_path)
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)

def load_kitti_bin_gt_7ch(file_path):
    # print(file_path)
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 7)

def kitti_to_dict(file_path, device, grid_size=0.05, segments=1):
    raw_data = load_kitti_bin(file_path)
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity as a feature

    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(intensity, dtype=torch.float32)
    ], dim=1).contiguous()

    num_points = features.shape[0]
    points_per_segment = (num_points + segments - 1) // segments

    batch_tensor = torch.arange(segments).repeat_interleave(points_per_segment)[:num_points]
    batch_tensor = batch_tensor.to(dtype=torch.int64)

    # print(batch_tensor)
    return {
        "coord": features[:, :3].contiguous().to(device),
        "feat": features.contiguous().to(device),
        "batch": batch_tensor.contiguous().to(device),
        "grid_size": torch.tensor(grid_size).to(device)
    }

def kitti_to_tensor(ndarray,device):
    return torch.tensor(ndarray).to(device)

def point_to_bin(output, output_path):
    # Convert to numpy and ensure it's on CPU
    a = output
    output = output.cpu().numpy()
    # Reshape to (N, 3) format for [x,y,z]
    points = output.reshape(-1, 3)
    # Save to binary file
    points.astype(np.float32).tofile(output_path)

class PointCloudDataset(Dataset):
    def __init__(self, file_paths, device, grid_size=0.05):
        self.file_paths = file_paths
        self.grid_size = grid_size
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # print(file_path)
        return kitti_to_dict(file_path, self.device)

class PointCloudGTDataset(Dataset):
    def __init__(self, file_paths, device ,grid_size=0.05):
        self.file_paths = file_paths
        self.grid_size = grid_size
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # print(file_path)
        gt = load_kitti_bin_gt_7ch(file_path)
        gt = np.delete(gt, 3, axis=1)

        gt = kitti_to_tensor(gt, self.device)
        return gt