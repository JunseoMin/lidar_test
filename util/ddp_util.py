import torch
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from .trainutil import PointCloudDataset, PointCloudGTDataset

from geomloss import SamplesLoss
from einops import rearrange

def train_ddp(
    local_rank,
    world_size,
    args,
    train_files,
    gt_files,
    model,
    creterion,
    do_validation=False,
    start_epoch = 1,
    num_epochs = 120,
    ckpt_best = "best_encoding",
    ckpt_latest = "latest_encoding",
    val_files = None,
    val_gt_files = None
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

    model = model.to(device)

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
    train_files_per_rank = train_files[local_rank::world_size]
    gt_files_per_rank    = gt_files[local_rank::world_size]

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
    # 5.1) Initialize wandb on rank=0
    ################################
    if local_rank == 0:
        wandb.init(
            project="lidar-encoding",  # Your W&B project name
            name="LiDAR-Encoder",            # A name for this run
            config={
                "learning_rate": 2e-4,
                "weight_decay": 1e-3,
                "epochs": 120,
                "batch_split": f"{len(train_files)} total / {world_size} ranks",
            }
        )
        # If you want gradient logging, you can optionally do:
        wandb.watch(ddp_model, log="gradients", log_freq=100)

    ################################
    # 6) Start training
    ################################
    for epoch in range(start_epoch, num_epochs + 1):
        # 6.1) Train
        ddp_model.train()
        total_loss = 0.0
        start_time = time.time()

        if local_rank == 0:
            print(f"----- Epoch {epoch} start (rank=0)-----")

        data_loader = zip(train_dataset, gt_dataset)
        skipped = 0
        num_samples_this_rank = 0

        for lidar_16, static_objects in data_loader:
            if static_objects is None or static_objects.size(0) == 0:
                skipped += 1
                continue

            optimizer.zero_grad()
            loss = creterion(static_objects, lidar_16, device=device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_samples_this_rank += 1

        # Gather total_loss from all ranks for global average
        total_loss_tensor = torch.tensor([total_loss, num_samples_this_rank], dtype=torch.float32, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)  # sum across ranks
        total_loss_all = total_loss_tensor[0].item()
        total_count_all = total_loss_tensor[1].item()
        avg_loss = total_loss_all / max(1e-8, total_count_all)

        scheduler.step()
        epoch_time = time.time() - start_time

        # Print/log only on rank 0
        if local_rank == 0:
            print(f"[Rank 0] Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Skipped: {skipped}, LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"Epoch time: {epoch_time:.2f} seconds")

            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "skipped_batches": skipped,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_time_sec": epoch_time,
            }, step=epoch)

        # 6.2) Validation
        # For demonstration, let's only do validation on rank 0
        # and then optionally all_reduce the result if each rank also did eval
        if local_rank == 0:
            logdict ={
                "epoch": epoch
            }

            if do_validation:
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

                # This code only runs in rank=0, so if you want truly distributed
                # evaluation, you would replicate the pattern used for training loss.
                if val_samples_count > 0:
                    avg_val_loss = val_total_loss / val_samples_count
                else:
                    avg_val_loss = 0.0

                print(f"[Rank 0] Validation done. Avg validation loss: {avg_val_loss:.6f}")
                
                logdict = {
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                }
                # Log validation metrics


            wandb.log(logdict, step=epoch)

            # 6.3) Save checkpoint only on rank 0
            if avg_loss < min_loss:
                if avg_loss < 0:
                    print("[Rank 0] ERROR: Negative loss, skip saving.")
                else:
                    min_loss = avg_loss
                    save_path = f"/home/server01/js_ws/lidar_test/ckpt/" + ckpt_best +".pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ddp_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'min_loss': min_loss
                    }, save_path)
                    print(f"[Rank 0] Best model saved at {save_path} with loss: {min_loss:.4f}")

            # Save "latest" checkpoint
            save_path = f"/home/server01/js_ws/lidar_test/ckpt/" + ckpt_latest +".pth"
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
        wandb.finish()  # End the wandb run
