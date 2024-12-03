import torch
import torch.nn as nn
from torch.utils.data import Dataset
from chamferdist import ChamferDistance

from model.LidarUpsample import Lidar4US

from geomloss import SamplesLoss

import time
import argparse
import glob
import numpy as np

from tqdm import tqdm

# from sklearn.neighbors import NearestNeighbors

def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def kitti_to_dict(file_path, grid_size=0.05, segments=1):
    raw_data = load_kitti_bin(file_path)
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity as a feature

    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(intensity, dtype=torch.float32)
    ], dim=1).contiguous()

    num_points = features.shape[0]
    segments = segments

    points_per_segment = (num_points + segments - 1) // segments

    batch_tensor = torch.arange(segments).repeat_interleave(points_per_segment)[:num_points]
    batch_tensor = batch_tensor.to(dtype=torch.int64)
    # batch_tensor = torch.full((features.shape[0],), 0, dtype=torch.int64)

    # print(batch_tensor)
    return {
        "coord": features[:, :3].contiguous().to("cuda"),
        "feat": features.contiguous().to("cuda"),
        "batch": batch_tensor.contiguous().to("cuda"),
        "grid_size": torch.tensor(grid_size).to("cuda")
    }

def kitti_to_tensor(file_path):
    return torch.tensor(load_kitti_bin(file_path)).to("cuda")


class PointCloudDataset(Dataset):
    def __init__(self, file_paths, grid_size=0.05):
        self.file_paths = file_paths
        self.grid_size = grid_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        return kitti_to_dict(file_path)


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.chamfer = ChamferDistance()
        self.emd = SamplesLoss(loss="sinkhorn", p=1, blur=0.05)
        self.emd_loss = 0
        self.chamfer_loss = 0
        
    def forward(self, pred_points, gt_points):
        # Make tensors contiguous
        pred_xyz = pred_points["feat"][:, :3].contiguous()
        gt_xyz = gt_points["feat"][:, :3].contiguous()
        
        # Reshape for Chamfer Distance
        pred_chamfer = pred_xyz.unsqueeze(0).contiguous()
        gt_chamfer = gt_xyz.unsqueeze(0).contiguous()
        
        # Compute both losses
        chamfer_loss = self.chamfer(gt_chamfer, pred_chamfer, bidirectional=False, point_reduction="mean")
        emd_loss = self.emd(pred_xyz, gt_xyz)
        
        # Combine losses
        total_loss = self.alpha * chamfer_loss + (1 - self.alpha) * emd_loss
        self.emd_loss = emd_loss
        self.chamfer_loss = chamfer_loss
        
        return total_loss


def train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, start_epoch=1, min_loss=float('inf'), num_epochs=120):
    print("========== train start ==========")
    model.to(device)
    model.train()

    for epoch in range(start_epoch, num_epochs + 1):
        total_loss = 0
        emd_avg = 0
        cd_avg = 0
        start_time = time.time()
        print(f"----- Epoch {epoch} start -----")
        
        data_loader = tqdm(zip(train_dataset, gt_dataset), total=len(train_dataset), desc=f"Epoch {epoch}/{num_epochs}")

        for train_data, gt_data in data_loader:
            
            # train_data = {key: value.to(device) for key, value in train_data.items()}
            # gt_data = {key: value.to(device) for key, value in gt_data.items()}
            
            optimizer.zero_grad()
            pred = model(train_data)  # Forward pass
            loss = criterion(pred, gt_data)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            total_loss += loss.item()
            emd_avg += criterion.emd_loss.item()
            cd_avg += criterion.chamfer_loss.item()

        avg_loss = total_loss / len(train_dataset)
        emd_avg /= len(train_dataset)
        cd_avg /= len(train_dataset)
        
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, CD: {cd_avg:.4f}, EMD: {emd_avg:.4f}")        
        print(f"Min Loss: {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")

        if avg_loss < min_loss:
            if avg_loss < 0:
                print(f"ERROR: Epoch {epoch} loss is negative: {avg_loss:.4f}")
                scheduler.step()
                print(f"================================")
                continue
            
            min_loss = avg_loss
            save_path = "/home/server01/js_ws/lidar_test/ckpt/best_model_v3.5.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
            print(f"Best model saved at {save_path} with loss: {min_loss:.4f}")

        save_path = f"/home/server01/js_ws/lidar_test/ckpt/latest_v3.5.pth"
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
        print(f"Model saved at {save_path}")

        scheduler.step()
        print(f"================================")
    print("========== train complete ==========")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Lidar4US(
        in_channels=4,  # coord + intensity
        drop_path=0.3,
        block_depth=(2, 2, 2, 4, 2),
        enc_channels=(32, 64, 128, 512, 512),
        enc_n_heads=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        stride=(2, 2, 2, 2),
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.1,
        mlp_ratio=4,
        dec_depths=(2, 2, 2, 2, 2),
        dec_n_head=(2, 4, 8, 16, 32 ),
        dec_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_channels=(128, 128, 128, 256, 512),
        train_decoder=True,
        exp_hidden=128,
        exp_out=128,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        upsample_ratio=4,
        out_channel=3,
    )
    
    # Define paths
    train_file_paths = glob.glob("/home/server01/js_ws/dataset/vertical_downsampled/train/**/*.bin", recursive=True)
    gt_file_paths = glob.glob("/home/server01/js_ws/dataset/vertical_downsampled/train_GT/**/*.bin", recursive=True)

    # Initialize dataset and dataloaders
    train_dataset = PointCloudDataset(train_file_paths)
    gt_dataset = PointCloudDataset(gt_file_paths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 80, 100], gamma=0.5) # 0.0004 0.0002 0.0001 0.00005
    criterion = HybridLoss(alpha=0.3)

    if args.resume_from:
        ckptr = torch.load(args.resume_from, map_location=device)   # load to device(GPU)
        model.load_state_dict(ckptr['model_state_dict'])
        model.to(device)
        scheduler.load_state_dict(ckptr['scheduler_state_dict'])
        optimizer.load_state_dict(ckptr['optimizer_state_dict'])
        start_epoch = ckptr['epoch'] + 1
        min_loss = ckptr['min_loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with min loss {min_loss:.4f}")
    else:
        start_epoch = 1
        min_loss = float('inf')
        print("No checkpoint found. Starting training from scratch.")

    train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, start_epoch=start_epoch, min_loss=min_loss, num_epochs=1200)
