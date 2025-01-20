import torch
import torch.nn.functional as F
import torch.optim as optim

from model import LiDARDiffusion

import argparse
import glob
import pyfiglet

from util import *

from geomloss import SamplesLoss    # use when validation
from tqdm import tqdm
from einops import rearrange

import time
import wandb

def train_diffusion(model, train_dataset, gt_dataset, device_train,
                    scheduler, optimizer, device_validation, validation_dataset,
                    validation_dataset_gt, min_loss = float('inf'), start_epoch = 0
                    ,num_epochs=120):
    """
    Args:
        model: LiDARDiffusion(encoder+diffusion) 통합 모델
        dataloader: static_objects, lidar_16 등을 (B,N,3)/(B,N,4) 형태로 받아오는 iterable
    """
    model.to(device_train)
    model.train()
    for epoch in range(start_epoch, num_epochs + 1):
        total_loss = 0.0
        start_time = time.time()

        print(f"----- Epoch {epoch} start -----")
        
        data_loader = tqdm(zip(train_dataset, gt_dataset),
                           total=len(train_dataset),
                           desc=f"Epoch {epoch}/{num_epochs}")
        skipped = 0
        for lidar_16, static_objects in data_loader:
            if static_objects is None or static_objects.size(0) == 0:
                skipped += 1
                continue
            
            optimizer.zero_grad()

            static_objects = static_objects
            lidar_16 = lidar_16

            loss = model.get_loss(static_objects, lidar_16, device_train)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)

        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
        print(f"Min Loss: {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")

        wandb.log({"train_loss": avg_loss, "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0]})

        # if not epoch % 10:
        #     print(f"----- validation start -----")
        #     val_total_loss = 0.0
        #     val_samples_count = 0
        #     model.eval()
        #     val_criterion = SamplesLoss(loss='sinkhorn', p=2, blur=.001, reach=.2)

        #     with torch.no_grad():
        #        for val_16, gt_val in zip(validation_dataset, validation_dataset_gt):
        #             if gt_val is None or gt_val.size(0) == 0:
        #                 continue
                    
        #             reconst = model.sample(val_16["feat"].shape[0], val_16, device_train)
        #             reconst = rearrange(reconst, "b n d -> (b n) d")
        #             val_loss = val_criterion(reconst, gt_val)
        #             val_total_loss += val_loss.item()
        #             val_samples_count += 1
            
        #     if val_samples_count > 0:
        #         avg_val_loss = val_total_loss / val_samples_count
        #     else:
        #         avg_val_loss = 0.0

        #     print(f"[INFO] Validation done. Avg validation loss: {avg_val_loss:.6f}")
        #     model.train()

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
        print(f"Model saved at {save_path}, skipped: {skipped}")
        scheduler.step()
        print(f"================================")

    print("========== train complete ==========")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from')

    wandb.init(project="lidar_diffusion_training", name="LiDARDiffusion")

    args = parser.parse_args()

    # Define paths
    train_file_paths = "/home/server01/js_ws/dataset/reconstruction_dataset/reconstruction_input/train/velodyne/"
    gt_file_paths = "/home/server01/js_ws/dataset/reconstruction_dataset/reconst_gt/train/"
    validation_file_paths = "/home/server01/js_ws/dataset/reconstruction_dataset/reconstruction_input/validation/velodyne/00/*.bin"
    validation_gt_file_paths = "/home/server01/js_ws/dataset/reconstruction_dataset/reconst_gt/validation/00/*.bin"

    train_dict = {}
    gt_dict = {}

    device_train = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_validation = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device_train)
    
    print("train device: ", device_train)
    print("loss device: ", device_validation)

    model = LiDARDiffusion(
        condition_drop_path = 0.3, 
        condition_enc_block_depth = (1, 1, 2), 
        condition_enc_channels = (8, 16, 32), 
        condition_enc_n_heads = (2, 4, 4),
        condition_enc_patch_size = (1024, 1024, 1024), 
        condition_qkv_bias = True, 
        condition_qk_scale = None, 
        condition_attn_drop  = 0.1, 
        condition_proj_drop = 0.1, 
        condition_mlp_ratio = 4, 
        condition_stride = (2, 2), 
        condition_in_channels = 4,
        condition_out_channel = 3,
        condition_hidden_channel = 8,
        drop_path = 0.3,
        enc_block_depth = (2, 2, 2, 3, 2),
        enc_channels = (16, 32, 64, 128, 256),
        enc_n_heads = (2, 2, 4, 4, 16),
        enc_patch_size = (1024, 1024, 1024, 1024, 1024), 
        qkv_bias = True, 
        qk_scale = None, 
        attn_drop = 0.1, 
        proj_drop = 0.1, 
        mlp_ratio = 4, 
        stride = (2, 2, 2, 2),
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        dec_depths = (2, 2, 2, 2),
        dec_channels = (16, 32, 64, 128),
        dec_n_head = (2, 4, 8, 8),
        dec_patch_size = (1024, 1024, 1024, 1024),
        time_out_ch = 3,
        num_steps = 5000,
        beta_1 = 1e-6,
        beta_T = 1e-2,
        device=device_train
    )

    train = []; gt = []; validation = []; validation_gt =[]

    for seq in range(2,11): # train for seq 02 to seq 11
        train += glob.glob(train_file_paths + f"{seq:02d}/*.bin")
        gt += glob.glob(gt_file_paths + f"{seq:02d}/*.bin")
    
    validation = glob.glob(validation_file_paths)
    validation_gt = glob.glob(validation_gt_file_paths)
    
    validation = validation[2:3]
    validation_gt = validation_gt[2:3]

    train_dict['train'] = train
    gt_dict['gt'] = gt

    print(train[10], train[20])
    ascii_art = pyfiglet.figlet_format("LIDAR DIFFUSION")
    print(ascii_art)

    # Initialize dataset and dataloaders for each sequence
    train_dataset = PointCloudDataset(train_dict['train'], device_train)
    gt_dataset = PointCloudGTDataset(gt_dict['gt'], device_train)
    
    validation_dataset = PointCloudDataset(validation, device_train)
    validation_gt_dataset = PointCloudGTDataset(validation_gt, device_train)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.5)

    if args.resume_from:
        ckptr = torch.load(args.resume_from, map_location=device_train)
        
        model.load_state_dict(ckptr['model_state_dict'])
        model.to(device_train)
        optimizer.load_state_dict(ckptr['optimizer_state_dict'])
        start_epoch = ckptr['epoch'] + 1
        scheduler.load_state_dict(ckptr['scheduler_state_dict'])
        min_loss = ckptr['min_loss']
        
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with min loss {min_loss:.4f}")
    else:
        start_epoch = 1
        min_loss = float('inf')
        print(f"Starting training for sequence {seq} from scratch.")

    train_diffusion(model=model, train_dataset=train_dataset, gt_dataset=gt_dataset, optimizer=optimizer,
                     scheduler=scheduler, device_train=device_train, device_validation=device_train,
                     validation_dataset=validation_dataset, validation_dataset_gt=validation_gt_dataset,
                     min_loss= min_loss, start_epoch=start_epoch, num_epochs=120)
    
    print(train[10], train[20])

