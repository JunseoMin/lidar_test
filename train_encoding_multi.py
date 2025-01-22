import os
import argparse
import time
import glob

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
import pyfiglet

from geomloss import SamplesLoss

# Import your modules
# from model.LidarEncoder import PTEncoder
# from util import PointCloudDataset, PointCloudGTDataset
from model.LidarEncoder import PTEncoder
from util import PointCloudDataset, PointCloudGTDataset
from datetime import timedelta


def train_loop_ddp(
    local_rank,
    world_size,
    args,
    train_files,
    gt_files,
    start_epoch=1,
    resume_min_loss=float('inf')
):
    """
    Main worker function for DDP training on a single GPU (local_rank).
    """
    # 1) Init process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://", 
        rank=local_rank, 
        world_size=world_size,
        timeout=timedelta(seconds=3600)
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 2) Build your model
    model = PTEncoder(
        in_channels = 4,
        drop_path = 0.3,
        enc_depths = (2, 2, 4, 4, 4, 4, 4, 2, 2),
        enc_channels = (32, 64, 128, 256, 512, 256, 128, 64, 32),
        enc_num_head = (2, 4, 8, 16, 32, 16, 8, 4, 2),
        enc_patch_size = (1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024), 
        qkv_bias = True, 
        qk_scale = None, 
        attn_drop = 0.1, 
        proj_drop = 0.1, 
        mlp_ratio = 4, 
        stride = (2, 2, 2, 2, 2, 2, 2, 2),
        order=("z", "z-trans", "hilbert", "hilbert-trans")
    ).to(device)

    # 3) Wrap model with DDP
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 4) Create optimizer, scheduler
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.5)

    # 5) Optionally resume from checkpoint
    min_loss = resume_min_loss
    if args.resume_from is not None and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        ddp_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        if local_rank == 0:
            print(f"[Rank 0] Resumed from epoch {start_epoch} with min loss {min_loss:.4f}")

    # 6) Slice data so each rank sees a different chunk
    train_files_per_rank = train_files[local_rank::world_size]
    gt_files_per_rank    = gt_files[local_rank::world_size]

    # Build datasets from the slices
    train_dataset = PointCloudDataset(train_files_per_rank, device, grid_size=0.01)
    gt_dataset    = PointCloudGTDataset(gt_files_per_rank, device)

    # 7) Initialize wandb only on rank=0
    if local_rank == 0:
        ascii_art = pyfiglet.figlet_format("LIDAR-CLUSTER-ENCODING")
        print(ascii_art)
        wandb.init(
            project="lidar_encoding_training", 
            name="LiDARENCODING-DDP",
            config={
                "learning_rate": 2e-4,
                "weight_decay": 1e-3,
                "epochs": 120,
                "train_files_total": len(train_files),
                "world_size": world_size,
            }
        )

    # 8) Define criterion
    criterion = SamplesLoss('sinkhorn', p=2, blur=0.001, scaling=0.99, debias=True, reach=.2)

    # 9) Training loop
    num_epochs = 120
    for epoch in range(start_epoch, num_epochs + 1):
        ddp_model.train()
        total_loss = 0.0
        skipped = 0

        start_time = time.time()

        if local_rank == 0:
            print(f"----- Epoch {epoch} start (rank=0)-----")

        # We'll zip across these smaller, per-rank datasets
        data_loader = zip(train_dataset, gt_dataset)
        num_samples_this_rank = 0

        for train_data, gt_data in data_loader:
            if gt_data is None or gt_data.size(0) == 0:
                skipped += 1
                continue

            optimizer.zero_grad()
            # Forward pass
            pred = ddp_model(train_data)  # ddp_model(...) calls model(...)

            # Compute loss
            loss = criterion(pred, gt_data)

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_samples_this_rank += 1

        # 9.1) Gather total_loss from all ranks
        total_loss_tensor = torch.tensor([total_loss, num_samples_this_rank], dtype=torch.float32, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)  # sum across ranks
        total_loss_all = total_loss_tensor[0].item()
        total_count_all = total_loss_tensor[1].item()
        avg_loss = total_loss_all / max(total_count_all, 1.0)

        scheduler.step()
        epoch_time = time.time() - start_time

        # 9.2) Print + wandb.log only on rank 0
        if local_rank == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Skipped: {skipped}, LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"Epoch {epoch} time: {epoch_time:.2f} seconds")

            wandb.log({
                "train_loss": avg_loss,
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_time": epoch_time
            })

            # 9.3) Save best / latest checkpoints
            if avg_loss < min_loss:
                if avg_loss < 0:
                    print(f"ERROR: negative loss, skip saving.")
                else:
                    min_loss = avg_loss
                    best_ckpt_path = "/home/server01/js_ws/lidar_test/ckpt/best_model_encoding_ddp.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ddp_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'min_loss': min_loss
                    }, best_ckpt_path)
                    print(f"Best model saved at {best_ckpt_path} with loss: {min_loss:.4f}")

            latest_ckpt_path = "/home/server01/js_ws/lidar_test/ckpt/latest_encoding_ddp.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, latest_ckpt_path)
            print(f"Model saved at {latest_ckpt_path}")
            print("================================")

    # 10) Cleanup
    dist.destroy_process_group()
    if local_rank == 0:
        print("========== Train complete ==========")
        wandb.finish()


def main():
    """Entry point for launching with multiple processes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', default=None, help='Path to checkpoint')
    args = parser.parse_args()

    # If using torchrun --nproc_per_node=NUM, PyTorch sets these environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Prepare file lists
    train_file_paths = "/home/server01/js_ws/dataset/reconstruction_dataset/reconstruction_input/train/velodyne/"
    gt_file_paths = "/home/server01/js_ws/dataset/odometry_dataset/encoding_gt/"

    train = []
    gt = []
    for seq in range(2, 11):
        train += glob.glob(train_file_paths + f"{seq:02d}/*.bin")
        gt += glob.glob(gt_file_paths + f"{seq:02d}/*.bin")

    # Start from 1, or from checkpoint
    start_epoch = 1
    min_loss = float('inf')

    # Now, each rank calls our worker function
    train_loop_ddp(
        local_rank=local_rank,
        world_size=world_size,
        args=args,
        train_files=train,
        gt_files=gt,
        start_epoch=start_epoch,
        resume_min_loss=min_loss
    )


if __name__ == "__main__":
    main()
