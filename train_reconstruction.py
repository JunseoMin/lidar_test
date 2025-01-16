import torch
import torch.nn as nn
from torch.utils.data import Dataset

from model.LidarUpsample import Lidar4US

from geomloss import SamplesLoss

import time
import argparse
import glob
import numpy as np

from tqdm import tqdm

import pyfiglet

def load_kitti_bin(file_path):
    # print(file_path)
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def load_kitti_bin_gt(file_path):
    # print(file_path)
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)

def kitti_to_dict(file_path, device, grid_size=0.05, segments=1):
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
        "coord": features[:, :3].contiguous().to(device),
        "feat": features.contiguous().to(device),
        "batch": batch_tensor.contiguous().to(device),
        "grid_size": torch.tensor(grid_size).to(device)
    }

def kitti_to_tensor(ndarray,device):
    return torch.tensor(ndarray).to(device)

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
        gt = load_kitti_bin_gt(file_path)
        gt = kitti_to_tensor(gt,self.device)
        return gt


def train_model_per_sequence(
    model, train_dataset, gt_dataset,
    optimizer, scheduler, criterion,
    device_train, device_loss,
    start_epoch=1, min_loss=float('inf'), num_epochs=120
):
    print("========== train start ==========")
    
    model = model.to(device_train)
    model.train()

    for epoch in range(start_epoch, num_epochs + 1):
        
        total_loss = 0.0
        start_time = time.time()
        print(f"----- Epoch {epoch} start -----")
        
        data_loader = tqdm(zip(train_dataset, gt_dataset),
                           total=len(train_dataset),
                           desc=f"Epoch {epoch}/{num_epochs}")
        skiped = 0

        for train_data, gt_data in data_loader:
            # Skip if no GT
            if gt_data is None or gt_data.size(0) == 0:
                skiped += 1
                continue

            optimizer.zero_grad()

            # 1) Forward on deviceTrain
            #    => train_data에는 coord, feat 등이 deviceTrain에 있어야 함
            #    => spconv 등 내부 연산도 deviceTrain에서 진행
            pred = model(train_data)


            if torch.isnan(pred["feat"]).any():
                print("[WARN]: model output has NAN, skip")
                print(pred.feat.item())                
                pred = None
                # Optional: this can help with fragmentation but will slow code
                skiped += 1
                return
            
            # 2) pred["feat"]를 deviceLoss로 복사 (autograd 그래프 유지를 위해 detach()는 사용하지 않음)
            pred_feat_loss = pred["feat"].to(device_loss, non_blocking=True)
            gt_data_loss   = gt_data.to(device_loss, non_blocking=True)

            # 3) Loss 계산은 deviceLoss에서
            #    (SamplesLoss도 deviceLoss에 생성되어 있어야 함)
            with torch.cuda.device(device_loss):
                loss = criterion(pred_feat_loss, gt_data_loss)

            # 4) Loss를 다시 deviceTrain으로 (gradient 역전파는 모델이 있는 deviceTrain에서)
            loss_train = loss.to(device_train, non_blocking=True)

            # 5) Backward in deviceTrain
            loss_train.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
        print(f"Min Loss: {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")

        # Update best loss
        if avg_loss < min_loss:
            if avg_loss < 0:
                print(f"ERROR: negative loss, skip saving.")
            else:
                min_loss = avg_loss
                save_path = f"/home/server01/js_ws/lidar_test/ckpt/best_model_reconst.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'min_loss': min_loss
                }, save_path)
                print(f"Best model saved at {save_path} with loss: {min_loss:.4f}")

        # Save latest
        save_path = f"/home/server01/js_ws/lidar_test/ckpt/latest_reconst.pth"
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
        print(f"Model saved at {save_path}, skiped: {skiped}")
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
        dec_depths=(2, 2, 2, 2),
        dec_n_head=(2, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_channels=(32, 64, 128, 256),
        train_decoder=True,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        out_channel = 3,
        fc_hidden = 32,
    )
    
    # Define paths
    train_file_paths = f"/home/server01/js_ws/dataset/reconstruction_input/train/velodyne/"
    gt_file_paths = f"/home/server01/js_ws/dataset/odometry_dataset/reconstruction_gt/"

    train_dict = {}
    gt_dict = {}

    device_train = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_loss = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device_train)
    
    print("train device: ", device_train)
    print("loss device: ", device_loss)

    train = []; gt = []

    for seq in range(2,11): # train for seq 2 to seq 11
        train += glob.glob(train_file_paths + f"{seq:02d}/*.bin")
        gt += glob.glob(gt_file_paths + f"{seq:02d}/*.bin")
    
    train_dict['train'] = train
    gt_dict['gt'] = gt

    print(train[10], train[20])

    ascii_art = pyfiglet.figlet_format("Hello World!")
    print(ascii_art)

    # Initialize dataset and dataloaders for each sequence
    train_dataset = PointCloudDataset(train_dict['train'], device_train)
    gt_dataset = PointCloudGTDataset(gt_dict['gt'], device_train)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    criterion = SamplesLoss(loss='sinkhorn', p=2, blur=.001, reach=.2)

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

    train_model_per_sequence(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device_train, device_loss, start_epoch=start_epoch, num_epochs=120)
    print(train[10], train[20])
