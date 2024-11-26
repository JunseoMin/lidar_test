from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from model.LidarUpsample import Lidar4US


def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def kitti_to_dict(file_path, grid_size=0.05, batch_id=0):
    raw_data = load_kitti_bin(file_path)
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity as a feature

    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(intensity, dtype=torch.float32)
    ], dim=1)

    batch_tensor = torch.full((features.shape[0],), 0, dtype=torch.int64)

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
        return kitti_to_dict(file_path, grid_size=self.grid_size, batch_id=idx)


class LidarUpsampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_points, gt_points):
        max_len = min(len(pred_points["feat"]), len(gt_points["feat"]))
        pred = pred_points["feat"][:max_len]
        gt = gt_points["coord"][:max_len]
        loss = self.criterion(pred, gt)
        return loss


def train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, num_epochs=120):
    print("========== train start ==========")
    model.to(device)
    model.train()

    min_loss = float('inf')  # Initialize the minimum loss

    for epoch in range(1, num_epochs + 1):
        total_loss = 0

        for train_data, gt_data in zip(train_dataset, gt_dataset):
            optimizer.zero_grad()
            pred = model(train_data)  # Forward pass
            loss = criterion(pred, gt_data)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save model if loss decreases
        if avg_loss < min_loss:
            min_loss = avg_loss
            save_path = "/home/server01/js_ws/lidar_test/ckpt/new_weights.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path} with loss: {min_loss:.4f}")
            
        # Update learning rate scheduler
        scheduler.step()

        save_path = "/home/server01/js_ws/lidar_test/ckpt/epoch_{epoch}.pth"
        torch.save(model.state_dict(), save_path)

        print("epoch {} finished",epoch)



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
    upsample_ratio=32,
    out_channel=3,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)    #0.001 0.0005

# StepLR Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

criterion = LidarUpsampleLoss()

# Train the model
# train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, num_epochs=120)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from glob import glob
from model.LidarUpsample import Lidar4US


def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def kitti_to_dict(file_path, grid_size=0.05, batch_id=0):
    raw_data = load_kitti_bin(file_path)
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity as a feature

    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(intensity, dtype=torch.float32)
    ], dim=1)

    batch_tensor = torch.full((features.shape[0],), 0, dtype=torch.int64)

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
        return kitti_to_dict(file_path, grid_size=self.grid_size, batch_id=idx)


class LidarUpsampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_points, gt_points):
        max_len = min(len(pred_points["feat"]), len(gt_points["feat"]))
        pred = pred_points["feat"][:max_len]
        gt = gt_points["coord"][:max_len]
        loss = self.criterion(pred, gt)
        return loss


def train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, num_epochs=120):
    print("========== train start ==========")
    model.to(device)
    model.train()

    min_loss = float('inf')  # Initialize the minimum loss

    for epoch in range(1, num_epochs + 1):
        total_loss = 0

        for train_data, gt_data in zip(train_dataset, gt_dataset):
            optimizer.zero_grad()
            pred = model(train_data)  # Forward pass
            loss = criterion(pred, gt_data)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
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

        print(f"epoch {epoch} finished")

# 기존 함수 및 클래스는 그대로 사용
# (load_kitti_bin, kitti_to_dict, PointCloudDataset, LidarUpsampleLoss)

def train_model(rank, world_size, train_file_paths, gt_file_paths):
    # 1. DDP 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 프로세스에 GPU를 고정
    device = torch.device(f"cuda:{rank}")

    # 2. 데이터셋 준비 및 분배
    train_dataset = PointCloudDataset(train_file_paths)
    gt_dataset = PointCloudDataset(gt_file_paths)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    gt_sampler = DistributedSampler(gt_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32, num_workers=4)
    gt_loader = DataLoader(gt_dataset, sampler=gt_sampler, batch_size=32, num_workers=4)

    # 3. 모델, 손실 함수, 옵티마이저 설정
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
    ).to(device)

    model = DDP(model, device_ids=[rank])  # DDP로 모델 감싸기
    criterion = LidarUpsampleLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 4. 학습 루프
    num_epochs = 120
    min_loss = float('inf')

    print(f"========== Train Start on Rank {rank} ==========")
    for epoch in range(1, num_epochs + 1):
        train_sampler.set_epoch(epoch)  # 샘플러에 에포크 설정
        gt_sampler.set_epoch(epoch)

        total_loss = 0

        for train_data, gt_data in zip(train_loader, gt_loader):
            optimizer.zero_grad()
            pred = model(train_data)  # Forward pass
            loss = criterion(pred, gt_data)  # Loss 계산
            loss.backward()  # Backpropagation
            optimizer.step()  # 매개변수 업데이트

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if rank == 0:  # Rank 0만 출력
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < min_loss and rank == 0:
            min_loss = avg_loss
            save_path = "/home/server01/js_ws/lidar_test/ckpt/new_weights.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path} with loss: {min_loss:.4f}")

        scheduler.step()

    dist.destroy_process_group()

# DistributedSampler 사용
train_sampler = DistributedSampler(train_dataset)
gt_sampler = DistributedSampler(gt_dataset)

# DataLoader 정의
train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # GPU당 처리할 배치 크기
    sampler=train_sampler,
    num_workers=16,  # 워커 프로세스 수
    pin_memory=True  # GPU로 데이터 전송 최적화
)

gt_loader = DataLoader(
    gt_dataset,
    batch_size=16,
    sampler=gt_sampler,
    num_workers=16,
    pin_memory=True
)
import torch.distributed as dist

# 분산 환경 초기화
dist.init_process_group(
    backend="nccl",  # NVIDIA GPU 환경에서는 'nccl' 사용
    init_method="env://",  # 환경 변수 기반 초기화
    world_size=2,  # GPU 개수
    rank=rank      # 현재 프로세스의 rank
)


import time

start_time = time.time()
for batch_idx, data in enumerate(train_loader):
    if batch_idx == 10:  
        break
print(f"10 batches loaded in {time.time() - start_time:.2f} seconds")
