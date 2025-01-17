import os
import argparse
import time
import glob
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# ---------------
# Import your model definitions
# (LiDARDiffusion, etc.)
# from model import LiDARDiffusion
# from util import PointCloudDataset, PointCloudGTDataset
# ---------------

from model import *
from util import *

from geomloss import SamplesLoss

def train_diffusion_ddp(
    local_rank,
    world_size,
    args,
    train_files,
    gt_files,
    val_files,
    val_gt_files
):
    """Main training loop for a single GPU process (local_rank)."""

    ################################
    # 1) Initialize process group
    ################################
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=local_rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)

    ################################
    # 2) Build model and move to GPU
    ################################
    device = torch.device(f"cuda:{local_rank}")

    model = LiDARDiffusion(
        condition_drop_path = 0.3, 
        condition_enc_block_depth = (1, 1, 1), 
        condition_enc_channels = (8, 8, 8), 
        condition_enc_n_heads = (2, 4, 8),
        condition_enc_patch_size = (64, 64, 64), 
        condition_qkv_bias = True, 
        condition_qk_scale = None, 
        condition_attn_drop  = 0.1, 
        condition_proj_drop = 0.1, 
        condition_mlp_ratio = 4, 
        condition_stride = (2, 2, 2), 
        condition_in_channels = 4,
        condition_out_channel = 3,
        condition_hidden_channel = 32,
        drop_path = 0.3,
        enc_block_depth = (2, 2, 2, 2, 2),
        enc_channels = (8, 8, 8, 8, 8),
        enc_n_heads = (2, 4, 8, 8, 8),
        enc_patch_size = (128, 128, 128, 128, 128), 
        qkv_bias = True, 
        qk_scale = None, 
        attn_drop = 0.1, 
        proj_drop = 0.1, 
        mlp_ratio = 4, 
        stride = (2, 2, 2, 2),
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        dec_depths = (2, 2, 2, 2),
        dec_channels = (8, 8, 8, 8),
        dec_n_head = (2, 4, 8, 8),
        dec_patch_size = (128, 128, 128, 128),
        time_out_ch = 3,
        num_steps = 500,
        beta_1 = 1e-4,
        beta_T = 1e-2,
        device=device
    ).to(device)

    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    ################################
    # 3) Create Optimizer, Scheduler
    ################################
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

    ############################################
    # 4) Slice data for each rank (quick method)
    ############################################
    # In production, use DistributedSampler and DataLoader. For demonstration, we do simple slicing.
    # Ensure each rank sees a unique chunk of data.
    train_files_per_rank = train_files[local_rank::world_size]
    gt_files_per_rank    = gt_files[local_rank::world_size]

    # Similarly for validation
    val_files_per_rank = val_files[local_rank::world_size]
    val_gt_files_per_rank = val_gt_files[local_rank::world_size]

    # Now create dataset objects for these slices
    train_dataset = PointCloudDataset(train_files_per_rank, device=device)
    gt_dataset    = PointCloudGTDataset(gt_files_per_rank, device=device)

    val_dataset = PointCloudDataset(val_files_per_rank, device=device)
    val_gt_dataset = PointCloudGTDataset(val_gt_files_per_rank, device=device)

    ################################
    # 5) Optionally resume from ckpt
    ################################
    start_epoch = 1
    min_loss = float('inf')
    if args.resume_from is not None and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        ddp_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        if local_rank == 0:
            print(f"[Rank 0] Resumed from epoch {start_epoch} with min loss {min_loss:.4f}")

    ################################
    # 6) Start training
    ################################
    num_epochs = 120
    for epoch in range(start_epoch, num_epochs + 1):
        # 6.1) Train
        # [Important for DistributedSampler: sampler.set_epoch(epoch). We skip that since we do slicing]
        ddp_model.train()
        total_loss = 0.0
        start_time = time.time()

        if local_rank == 0:
            print(f"----- Epoch {epoch} start (rank=0)-----")

        # We just zip across sliced dataset
        # Each rank sees ~1/world_size portion
        # If one rank has fewer data points than another, it will finish earlier in the loop
        data_loader = zip(train_dataset, gt_dataset)
        skipped = 0
        num_samples_this_rank = 0

        for lidar_16, static_objects in data_loader:
            if static_objects is None or static_objects.size(0) == 0:
                skipped += 1
                continue

            optimizer.zero_grad()
            loss = ddp_model.module.get_loss(static_objects, lidar_16, device=device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_samples_this_rank += 1

        # We need to gather total_loss from all ranks, so we can compute an overall average.
        # For simplicity, you can reduce from all ranks to rank 0, and then compute average.
        total_loss_tensor = torch.tensor([total_loss, num_samples_this_rank], dtype=torch.float32, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)  # sum across ranks
        total_loss_all = total_loss_tensor[0].item()
        total_count_all = total_loss_tensor[1].item()
        avg_loss = total_loss_all / max(1e-8, total_count_all)

        # Print logs only on rank 0
        if local_rank == 0:
            print(f"[Rank 0] Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Skipped: {skipped}, LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"Epoch time: {time.time() - start_time:.2f} seconds")

        scheduler.step()

        # 6.2) Validation (optionally only do on rank 0 to reduce overhead)
        # If you want to average validation loss across all ranks, adapt similarly to training step above.
        # For demonstration, let's just do rank 0 validation:
        if local_rank == 0:
            if epoch:
                ddp_model.eval()
                val_total_loss = 0.0
                val_samples_count = 0
                val_criterion = SamplesLoss(loss='sinkhorn', p=2, blur=.001, reach=.2)

                with torch.no_grad():
                    for val_16, gt_val in zip(val_dataset, val_gt_dataset):
                        if gt_val is None or gt_val.size(0) == 0:
                            continue
                        
                        reconst = ddp_model.module.sample(50000, val_16, device)
                        reconst = rearrange(reconst, "b n d -> (b n) d")
                        val_loss = val_criterion(reconst, gt_val)
                        val_total_loss += val_loss.item()
                        val_samples_count += 1

                # 모든 rank의 validation loss 합산
                val_loss_tensor = torch.tensor([val_total_loss, val_samples_count], dtype=torch.float32, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)  # 모든 rank의 결과 합산
                total_val_loss = val_loss_tensor[0].item()
                total_val_count = val_loss_tensor[1].item()
                avg_val_loss = total_val_loss / max(1, total_val_count)

                if local_rank == 0:
                    print(f"[Rank 0] Validation done. Avg validation loss: {avg_val_loss:.6f}")

            # 6.3) Save checkpoint only on rank 0
            # Decide if it's best so far
            if avg_loss < min_loss:
                if avg_loss < 0:
                    print("[Rank 0] ERROR: Negative loss, skip saving.")
                else:
                    min_loss = avg_loss
                    save_path = f"/home/server01/js_ws/lidar_test/ckpt/best_model_reconst.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ddp_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'min_loss': min_loss
                    }, save_path)
                    print(f"[Rank 0] Best model saved at {save_path} with loss: {min_loss:.4f}")

            # Save "latest" checkpoint
            save_path = f"/home/server01/js_ws/lidar_test/ckpt/latest_reconst.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
            print(f"[Rank 0] Model saved at {save_path}")

    # End of training
    dist.destroy_process_group()
    if local_rank == 0:
        print("[Rank 0] Training complete.")


def main():
    """Entry point for launching with multiple processes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', default=None, help='path to checkpoint')
    args = parser.parse_args()

    # The number of processes/GPU cards you want to use
    # If using torchrun --nproc_per_node=NUM, you can read env variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ---------------
    # Prepare your file lists
    # ---------------
    train_file_paths = "/home/server01/js_ws/dataset/reconstruction_input/train/velodyne/"
    gt_file_paths = "/home/server01/js_ws/dataset/odometry_dataset/reconstruction_gt/"
    validation_file_paths = "/home/server01/js_ws/dataset/reconstruction_input/validation/velodyne/00/*.bin"
    validation_gt_file_paths = "/home/server01/js_ws/dataset/odometry_dataset/reconstruction_gt/00/*.bin"

    train_files = []
    gt_files    = []
    for seq in range(2,11):
        train_files += glob.glob(train_file_paths + f"{seq:02d}/*.bin")
        gt_files    += glob.glob(gt_file_paths    + f"{seq:02d}/*.bin")

    # We’ll just pick some validation subset
    val_files = glob.glob(validation_file_paths)[:10]
    val_gt_files = glob.glob(validation_gt_file_paths)[:10]

    # For a quick test, let's slice them even smaller
    train_files = train_files  # e.g. 20 samples
    gt_files = gt_files

    # Now, if using torchrun or `torch.distributed.launch`, each rank calls main() with a different local_rank
    train_diffusion_ddp(
        local_rank=local_rank,
        world_size=world_size,
        args=args,
        train_files=train_files,
        gt_files=gt_files,
        val_files=val_files,
        val_gt_files=val_gt_files
    )


if __name__ == "__main__":
    main()
