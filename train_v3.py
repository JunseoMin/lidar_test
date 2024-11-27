import torch
import torch.nn as nn
from torch.utils.data import Dataset
from chamferdist import ChamferDistance

from model.LidarUpsample import Lidar4US

import time
import argparse
import glob
import numpy as np

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
        self.criterion = ChamferDistance()

    def forward(self, pred_points, gt_points):
        # Only use XYZ coordinates for distance calculation
        pred_xyz = pred_points["feat"][:]  # Take only XYZ coordinates
        gt_xyz = gt_points["feat"][:, :3]      # Take only XYZ coordinates

        # Reshape to (batch_size, num_points, 3)
        pred = pred_xyz.unsqueeze(0)
        gt = gt_xyz.unsqueeze(0)
        
        # Calculate bidirectional Chamfer Distance
        loss = self.criterion(pred, gt, bidirectional=True, point_reduction="mean")

        return loss


def train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, start_epoch=1, min_loss=float('inf'), num_epochs=120):
    print("========== train start ==========")
    model.to(device)
    model.train()

    for epoch in range(start_epoch, num_epochs + 1):
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
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Min Loss: {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")        
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")

        if avg_loss < min_loss:
            min_loss = avg_loss
            save_path = "/home/server01/js_ws/lidar_test/ckpt/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
            print(f"Best model saved at {save_path} with loss: {min_loss:.4f}")

        save_path = f"/home/server01/js_ws/lidar_test/ckpt/latest.pth"
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
        print(f"Model saved at {save_path}")

        scheduler.step()
        print(f"==== epoch {epoch} finished! ====")
    print("========== train complete ==========")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from')

    args = parser.parse_args()

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
    
    # Define paths
    train_file_paths = glob.glob("/home/server01/js_ws/dataset/sparse_pointclouds_kitti/train/*.bin")
    gt_file_paths = glob.glob("/home/server01/js_ws/dataset/sparse_pointclouds_kitti/GT/*.bin")

    # Initialize dataset and dataloaders
    train_dataset = PointCloudDataset(train_file_paths)
    gt_dataset = PointCloudDataset(gt_file_paths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = LidarUpsampleLoss()

    if args.resume_from:
        ckptr = torch.load(args.resume_from)
        model.load_state_dict(ckptr['model_state_dict'])
        scheduler.load_state_dict(ckptr['scheduler_state_dict'])
        optimizer.load_state_dict(ckptr['optimizer_state_dict'])
        start_epoch = ckptr['epoch'] + 1
        min_loss = ckptr['min_loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with min loss {min_loss:.4f}")
    else:
        start_epoch = 1
        min_loss = float('inf')
        print("No checkpoint found. Starting training from scratch.")

    train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, start_epoch=start_epoch, min_loss=min_loss, num_epochs=120)
