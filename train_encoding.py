from model.LidarEncoder import PTEncoder
from util import *

from torch.optim.lr_scheduler import ChainedScheduler

import glob
import time
from tqdm import tqdm

from geomloss import SamplesLoss
# import os
import pyfiglet
import wandb
import argparse

from .loss.CombinedLossAE import *

def train_encoding(model, train_dataset, gt_dataset, device_train,
                    scheduler, optimizer, criterion, min_loss = float('inf'), 
                    start_epoch = 0, num_epochs=150, train_decoder = True):
    model.to(device_train)
    model.train()

    for epoch in range(start_epoch, num_epochs + 1):
        total_loss = 0.
        skipped = 0

        train_decoder = False

        if epoch < 20:
            for param in model.decoder.parameters():
                param.requires_grad = False
            
            train_decoder = True
            

        start_time = time.time()
        print(f"----- Epoch {epoch} start -----")
        data_loader = tqdm(zip(train_dataset, gt_dataset), total=len(train_dataset), desc=f"Epoch {epoch}/{num_epochs}")

        for train_data, gt_data in data_loader:
        
            if gt_data is None or gt_data.size(0) == 0:
                skipped += 1
                continue
                        
            optimizer.zero_grad()
            pred_feat,pred_decoder = model(train_data, train_decoder)  # Forward pass
            loss = criterion(pred_feat = pred_feat, pred_decoder= pred_decoder,input_data = train_data["feat"], gt_data = gt_data)  # Compute loss
            # gt_data: Contains 100m centroids map
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            total_loss += loss.item()    
        avg_loss = total_loss / len(train_dataset)
        
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
        print(f"Min Loss: {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")
        wandb.log({"train_loss": avg_loss, "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0]})

        if avg_loss < min_loss:
            if avg_loss < 0:
                print(f"ERROR: negative loss, skip saving.")
            else:
                min_loss = avg_loss
                save_path = f"/home/server01/js_ws/lidar_test/ckpt/best_model_encoding.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'min_loss': min_loss
                }, save_path)
                print(f"Best model saved at {save_path} with loss: {min_loss:.4f}")

        # Save latest
        save_path = f"/home/server01/js_ws/lidar_test/ckpt/latest_encoding.pth"
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

parser = argparse.ArgumentParser()
parser.add_argument('--resume-from')
args = parser.parse_args()

device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("training device: ", device_train)
ascii_art = pyfiglet.figlet_format("LIDAR-CLUSTER-ENCODING")
print(ascii_art)

wandb.init(project="lidar_encoding_training", name="LiDARENCODING")

model = PTEncoder(
                 in_channels = 4,
                 drop_path = 0.3,
                 enc_depths = (2, 2, 2, 4, 2, 2, 2),
                 enc_channels = (32, 64, 128, 256, 512, 256, 128),
                 enc_num_head = (2, 4, 8, 16, 32, 16, 8),
                 enc_patch_size = (1024, 1024, 1024, 1024, 1024, 1024, 1024), 
                 qkv_bias = True, 
                 qk_scale = None, 
                 attn_drop = 0.1, 
                 proj_drop = 0.1, 
                 mlp_ratio = 4, 
                 stride = (2, 2, 2, 2, 2, 2),
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 out_channels=6,
                 dec_depths=(1, 1, 2, 2, 1, 1),
                 dec_channels=(4, 8, 16, 32, 64, 64),
                 dec_num_head=(2, 2, 4, 8, 8, 8),
                 dec_patch_size=(1024, 1024, 1024, 1024, 1024, 1024),
)

train = []; gt = []

train_file_paths = "/home/server01/js_ws/dataset/reconstruction_dataset/reconstruction_input/train/velodyne/"
gt_file_paths = "/home/server01/js_ws/dataset/encoder_dataset/encoder_xyzn"

for seq in range(2,11): # train for seq 02 to seq 11
    train += glob.glob(train_file_paths + f"{seq:02d}/*.bin")
    gt += glob.glob(gt_file_paths + f"{seq:02d}/*.bin")

train_dataset = PointCloudDataset(train, device_train, grid_size=0.01)  # 16channel lidar
gt_dataset = PointCloudGTDataset(gt, device_train)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-8, weight_decay=1e-5)

scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20], gamma=10)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,120], gamma=0.5)

scheduler = ChainedScheduler([scheduler1,scheduler2], optimizer)

criterion = CombinedCriterionAE()

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

train_encoding(model = model, train_dataset = train_dataset, gt_dataset = gt_dataset, device_train = device_train, scheduler = scheduler, 
               optimizer = optimizer, criterion = criterion, start_epoch=start_epoch, min_loss=min_loss)