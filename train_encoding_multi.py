import os
import argparse
import time
import glob

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from tqdm import tqdm
import wandb
import pyfiglet

# Import your modules (model, datasets, losses, etc.)
from model.LidarEncoder import PTEncoder
from util import PointCloudDataset, PointCloudGTDataset
from loss.CombinedLossAE import CombinedCriterionAE
from torch.optim.lr_scheduler import ChainedScheduler

# ------------------------------------------------------------------
# 1. Distributed Setup Utilities
# ------------------------------------------------------------------
def setup_for_distributed(rank: int, world_size: int):
    """
    Initialize the default process group.
    """
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    # Make sure each GPU prints only if rank == 0
    # or you can guard printing with if rank == 0
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


# ------------------------------------------------------------------
# 2. Distributed Training Loop
# ------------------------------------------------------------------
def train_encoding(rank, world_size, args):
    """
    rank: current process rank
    world_size: total number of processes (GPUs)
    args: parsed arguments (for e.g. resume_from, etc.)
    """
    # 2.1 Setup distributed environment
    setup_for_distributed(rank, world_size)
    
    # 2.2 Create device for this process
    device_train = torch.device("cuda", rank)

    # 2.3 Build model
    model = PTEncoder(
        in_channels=4,
        drop_path=0.3,
        enc_depths=(2, 2, 4, 6, 4, 2, 2),
        enc_channels=(32, 64, 128, 256, 512, 256, 128),
        enc_num_head=(2, 4, 8, 16, 32, 16, 8),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024, 1024, 1024),
        qkv_bias=True, 
        qk_scale=None, 
        attn_drop=0.1, 
        proj_drop=0.1, 
        mlp_ratio=4, 
        stride=(2, 2, 2, 2, 2, 2),
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        out_channels=6,
        dec_depths=(1, 1, 2, 2, 1, 1),
        dec_channels=(4, 8, 16, 32, 64, 64),
        dec_num_head=(2, 2, 4, 8, 8, 8),
        dec_patch_size=(1024, 1024, 1024, 1024, 1024, 1024),
    )
    model.to(device_train)

    # 2.4 Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # 2.5 Prepare dataset & data loader with Distributed Sampler
    train_file_paths = "/home/server01/js_ws/dataset/reconstruction_dataset/reconstruction_input/train/velodyne/"
    gt_file_paths = "/home/server01/js_ws/dataset/encoder_dataset/encoder_xyzn/"

    train_bin_files = []
    gt_bin_files = []
    for seq in range(2, 11):  # train for seq 02 to seq 11
        train_bin_files += glob.glob(train_file_paths + f"{seq:02d}/*.bin")
        gt_bin_files += glob.glob(gt_file_paths + f"{seq:02d}/*.bin")

    train_dataset = PointCloudDataset(train_bin_files, device_train, grid_size=0.01)
    gt_dataset = PointCloudGTDataset(gt_bin_files, device_train)

    # *Important*: Use DistributedSampler for each dataset
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    gt_sampler = DistributedSampler(gt_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=2)
    gt_loader = DataLoader(gt_dataset, batch_size=1, sampler=gt_sampler, num_workers=2)

    # 2.6 Setup optimizer, scheduler, criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-8, weight_decay=1e-5)
    
    # Example of a Chained Scheduler usage
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20], gamma=10)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,120], gamma=0.5)
    scheduler = ChainedScheduler([scheduler1, scheduler2], optimizer)

    criterion = CombinedCriterionAE()

    # 2.7 Optionally load checkpoint if resume_from is provided
    start_epoch = 1
    min_loss = float('inf')
    if args.resume_from is not None and os.path.exists(args.resume_from):
        ckptr = torch.load(args.resume_from, map_location=device_train)
        model.load_state_dict(ckptr['model_state_dict'])
        optimizer.load_state_dict(ckptr['optimizer_state_dict'])
        scheduler.load_state_dict(ckptr['scheduler_state_dict'])
        start_epoch = ckptr['epoch'] + 1
        min_loss = ckptr['min_loss']
        if rank == 0:
            print(f"[Rank 0] Resumed from epoch {start_epoch}, min_loss: {min_loss:.4f}")

    # 2.8 (Optional) Only rank 0 initializes wandb, prints fancy ascii, etc.
    if rank == 0:
        ascii_art = pyfiglet.figlet_format("LIDAR-CLUSTER-ENCODING")
        print(ascii_art)
        wandb.init(project="lidar_encoding_training", name="LiDARENCODING")

    # 2.9 Start training
    num_epochs = 150
    for epoch in range(start_epoch, num_epochs + 1):
        # set sampler epoch for shuffling
        train_sampler.set_epoch(epoch)
        gt_sampler.set_epoch(epoch)

        model.train()  # DDP model is set to train mode
        total_loss = 0.0
        rec_total = 0.0
        reg_total = 0.0
        norm_total = 0.0
        skipped = 0

        # decide if training the decoder
        # from your logic: self-supervised for first 20 epochs
        train_decoder = True if epoch < 20 else False
        
        # after 20 epoch: freeze decoder
        if epoch == 20:
            if rank == 0:
                print("[Rank 0] Switch to supervised: Freeze decoder, alpha=0, beta=1, gamma=1")
            for param in model.module.dec.parameters():
                param.requires_grad = False
            criterion.set_weights(alpha=0, beta=1, gamma=1)

        start_time = time.time()

        # Create a combined iterable
        data_loader = zip(train_loader, gt_loader)

        # You can use tqdm on rank=0 only (to avoid console spam)
        if rank == 0:
            data_loader = tqdm(data_loader, total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}")

        for (train_data, gt_data) in data_loader:
            # Check for invalid GT (just as in your code)
            if gt_data is None or gt_data.size(0) == 0:
                skipped += 1
                continue

            optimizer.zero_grad()

            # Forward pass
            pred_feat, pred_decoder = model(train_data, train_decoder)

            # Compute loss
            loss = criterion(
                pred_feat=pred_feat,
                pred_decoder=pred_decoder,
                input_data=train_data["feat"],
                gt_data=gt_data,
                train_decoder=train_decoder
            )

            # Backprop
            loss.backward()
            optimizer.step()

            # For logging: accumulate
            total_loss += loss.item()

            rec, reg, norm = criterion.get_loss()
            rec_total += rec
            reg_total += reg
            norm_total += norm

        # -- All ranks might want to reduce the total loss so we get the global average --
        # You can do something like:
        total_loss_tensor = torch.tensor([total_loss], device=device_train)
        rec_tensor = torch.tensor([rec_total], device=device_train)
        reg_tensor = torch.tensor([reg_total], device=device_train)
        norm_tensor = torch.tensor([norm_total], device=device_train)

        # Reduce sums to rank 0
        dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(rec_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(reg_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(norm_tensor, dst=0, op=dist.ReduceOp.SUM)

        # if rank=0, compute average
        if rank == 0:
            # average over the entire dataset size = len(train_loader)*world_size
            # but if your batch_size=1 or different, you can just do
            # average by total steps. The total steps are len(train_loader) on each GPU.
            steps_per_epoch = len(train_loader)
            
            avg_loss = total_loss_tensor.item() / steps_per_epoch
            avg_rec = rec_tensor.item() / steps_per_epoch
            avg_reg = reg_tensor.item() / steps_per_epoch
            avg_norm = norm_tensor.item() / steps_per_epoch

            current_lr = scheduler.get_last_lr()[0] if len(scheduler.get_last_lr()) > 0 else None

            print(f"[Rank 0] Epoch {epoch}/{num_epochs} => Loss: {avg_loss:.4f} | rec: {avg_rec:.4f}, reg: {avg_reg:.4f}, norm: {avg_norm:.4f}")
            print(f"[Rank 0] Min Loss: {min_loss:.4f}, LR: {current_lr}")
            print(f"[Rank 0] Skipped: {skipped}, epoch time: {time.time() - start_time:.2f}s")

            # wandb logging
            wandb.log({
                "train_loss": avg_loss,
                "epoch": epoch,
                "learning_rate": current_lr,
                "rec_loss": avg_rec,
                "reg_loss": avg_reg,
                "norm_loss": avg_norm
            })

            # Checkpoint saving logic
            if avg_loss < min_loss and avg_loss >= 0:
                min_loss = avg_loss
                save_path = f"/home/server01/js_ws/lidar_test/ckpt/best_model_encoding.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # note the .module
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'min_loss': min_loss
                }, save_path)
                print(f"[Rank 0] Best model saved at {save_path} with loss: {min_loss:.4f}")

            # Save latest
            save_path = f"/home/server01/js_ws/lidar_test/ckpt/latest_encoding.pth"
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # note the .module
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'min_loss': min_loss
                }, save_path)
            print(f"[Rank 0] Latest model saved at {save_path}")

        # Update scheduler
        scheduler.step()

        # Let all ranks sync here so next epoch starts in sync
        dist.barrier()
        
    if rank == 0:
        print("[Rank 0] Training complete.")

    # 2.10 Cleanup
    cleanup()


# ------------------------------------------------------------------
# 3. Entry point: spawn processes for each GPU
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', default=None, type=str, help="Path to checkpoint to resume from")
    # If you are using torchrun, you do NOT necessarily need to parse local_rank manually
    # but for launching via python -m torch.distributed.launch, you might:
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='number of GPUs')
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # If you are using `torchrun --nproc_per_node=world_size ddp_train.py`
    # Then `LOCAL_RANK` is automatically set. We'll fetch it in `train_encoding`.
    # We'll spawn processes if you prefer the mp.spawn approach. 
    # Option A: Using torch.multiprocessing.spawn
    # mp.spawn(train_encoding, nprocs=world_size, args=(world_size, args))
    
    # Option B: If you rely on torchrun with automatic rank assignment:
    #  you can fetch the rank from environ, then call train_encoding(rank, world_size, args)
    # This code uses Option B for simplicity:
    rank = int(os.environ.get("LOCAL_RANK", 0))
    train_encoding(rank, world_size, args)


if __name__ == "__main__":
    main()
