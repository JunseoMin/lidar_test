import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from model.LidarUpsample import Lidar4US

import time

def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def kitti_to_dict(file_path, grid_size=0.05, segments=3):
    raw_data = load_kitti_bin(file_path)
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity as a feature

    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(intensity, dtype=torch.float32)
    ], dim=1)

    num_points = features.shape[0]
    segments = segments

    points_per_segment = (num_points + segments - 1) // segments

    batch_tensor = torch.arange(segments).repeat_interleave(points_per_segment)[:num_points]
    batch_tensor = batch_tensor.to(dtype=torch.int64)
    # batch_tensor = torch.full((features.shape[0],), 0, dtype=torch.int64)

    # print(batch_tensor)
    return {
        "coord": features[:, :3].to("cuda"),
        "feat": features.to("cuda"),
        "batch": batch_tensor.to("cuda"),
        "grid_size": torch.tensor(grid_size).to("cuda")
    }


class PointCloudDataset(Dataset):
    def __init__(self, file_paths, grid_size=0.05):
        self.file_paths = file_paths
        self.grid_size = grid_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        return kitti_to_dict(file_path, grid_size=self.grid_size)


class LidarUpsampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_points, gt_points):
        max_len = min(len(pred_points["feat"]), len(gt_points["feat"][:,:3]))
        # print(max_len)
        # print("pred shape: ",pred_points["feat"].shape)
        # print(gt_points.shape)
        pred = pred_points["feat"][:max_len]
        gt = gt_points["feat"][:max_len,:3]
        loss = self.criterion(pred, gt)
        return loss


def train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, num_epochs=120):
    print("========== train start ==========")
    model.to(device)
    model.train()

    min_loss = float('inf')  # Initialize the minimum loss

    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        start_time = time.time()

        for train_data, gt_data in zip(train_dataset, gt_dataset):
            optimizer.zero_grad()
            pred = model(train_data)  # Forward pass
            loss = criterion(pred, gt_data)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Min Loss {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")        
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")
        # Save model if loss decreases
        if avg_loss < min_loss:
            min_loss = avg_loss
            save_path = "/home/server01/js_ws/lidar_test/ckpt/new_weights.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path} with loss: {min_loss:.4f}")
            
        # Update learning rate scheduler
        scheduler.step()

        save_path = f"/home/server01/js_ws/lidar_test/ckpt/epoch_{epoch}.pth"
        torch.save(model.state_dict(), save_path)

        print(f"==== epoch {epoch} finished ====")



# Define paths
train_file_paths = glob("/home/server01/js_ws/dataset/sparse_pointclouds_kitti/train/*.bin", recursive=True)
gt_file_paths = glob("/home/server01/js_ws/dataset/sparse_pointclouds_kitti/GT/*.bin")

# Initialize dataset and dataloaders
train_dataset = PointCloudDataset(train_file_paths)
gt_dataset = PointCloudDataset(gt_file_paths)

# Define model, optimizer, scheduler, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Lidar4US(
    in_channels=4,  # coord + intensity
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
    upsample_ratio=16,
    out_channel=3,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

# StepLR Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

criterion = LidarUpsampleLoss()

# Train the model
train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, num_epochs=60)
