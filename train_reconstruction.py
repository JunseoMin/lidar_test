import torch
import torch.nn as nn
from torch.utils.data import Dataset

from model.LidarUpsample import Lidar4US
from reconstruction.pruningloss import PruingLoss

import time
import argparse
import glob
import numpy as np

from tqdm import tqdm

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


def train_model_per_sequence(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, start_epoch=1, min_loss=float('inf'), num_epochs=120):
    print("========== train start ==========")
    model.to(device)
    model.train()
    for epoch in range(start_epoch, num_epochs + 1):
        
        total_loss = 0
        WD_avg = 0
        start_time = time.time()
        print(f"----- Epoch {epoch} start -----")
        
        data_loader = tqdm(zip(train_dataset, gt_dataset), total=len(train_dataset), desc=f"Epoch {epoch}/{num_epochs}")

        for train_data, gt_data in data_loader:
            optimizer.zero_grad()
            pred = model(train_data)  # Forward pass
            loss = criterion(pred["feat"].cpu(), gt_data["feat"].cpu())  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            total_loss += loss.item()
            WD_avg += criterion.WD_loss.item()

        avg_loss = total_loss / len(train_dataset)
        WD_avg /= len(train_dataset)
        
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, EMD: {WD_avg:.4f}")
        print(f"Min Loss: {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")

        if avg_loss < min_loss:
            if avg_loss < 0:
                print(f"ERROR: Epoch {epoch} loss is negative: {avg_loss:.4f}")
                scheduler.step()
                print(f"================================")
                continue
            
            min_loss = avg_loss
            save_path = f"/home/server01/js_ws/lidar_test/ckpt/best_model_vertical_upsample.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
            print(f"Best model saved at {save_path} with loss: {min_loss:.4f}")

        save_path = f"/home/server01/js_ws/lidar_test/ckpt/latest_vertical_upsample.pth"
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
        block_depth=(2, 2, 2, 2, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_n_heads=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        stride=(2, 2, 2, 2),
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.1,
        mlp_ratio=4,
        dec_depths=(2, 2, 2, 2, 2),
        dec_n_head=(2, 2, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_channels=(32, 64, 128, 256, 512),
        train_decoder=True,
        exp_hidden=128,
        exp_out=64,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        upsample_ratio=4,
        out_channel=3,
    )
    
    # Define paths
    train_file_paths = f"/home/server01/js_ws/dataset/odometry_dataset/train/"
    gt_file_paths = f"/home/server01/js_ws/dataset/odometry_dataset/gt/"
    map_file_paths = f"/home/server01/js_ws/dataset/odometry_dataset/gt_map/"

    train_dict = {}
    gt_dict = {}
    for seq in range(11):
        train = glob.glob(train_file_paths + f"{seq:02d}/*.bin")
        gt = glob.glob(gt_file_paths + f"{seq:02d}/*.bin")
        train_dict[seq] = train
        gt_dict[seq] = gt

    # Initialize dataset and dataloaders for each sequence
    for seq in range(4,5):
        train_dataset = PointCloudDataset(train_dict[seq])
        gt_dataset = PointCloudDataset(gt_dict[seq])
        
        map_file = load_kitti_bin(map_file_paths + f"{seq:02d}_map.bin")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.5)
        criterion = PruingLoss()
        criterion.set_map(map_file)

        if args.resume_from:
            ckptr = torch.load(args.resume_from, map_location=device)
            
            model.load_state_dict(ckptr['model_state_dict'])
            model.to(device)
            optimizer.load_state_dict(ckptr['optimizer_state_dict'])
            start_epoch = ckptr['epoch'] + 1
            scheduler.load_state_dict(ckptr['scheduler_state_dict'])
            min_loss = ckptr['min_loss']
            
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with min loss {min_loss:.4f}")
        else:
            start_epoch = 1
            min_loss = float('inf')
            print(f"Starting training for sequence {seq} from scratch.")

        train_model_per_sequence(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, start_epoch=start_epoch, num_epochs=120)
