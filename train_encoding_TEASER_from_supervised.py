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

from loss.CombinedLossAETeaser import *

import sys

def train_encoding(model, train_dataset, gt_dataset, device_train,
                   scheduler, optimizer, criterion, 
                   min_loss=float('inf'), start_epoch=1, num_epochs=150,
                   train_decoder=False,
                   start_flag=0):
    """
    Args:
      start_flag (int): 배치를 몇 개 스킵할 것인지(이전 checkpoint에서 저장된 flag)
    """
    model.to(device_train)
    model.train()
    print(f"Train self-supervised")

    for epoch in range(start_epoch, num_epochs + 1):
        total_loss = 0.
        skipped = 0
        flag = 0

        rec_total = 0.
        reg_total = 0.
        norm_total = 0.

        iteration_count = 0  # 이 epoch 내 배치 수
        train_decoder_local = train_decoder

        # if epoch <= 20:  # train self-supervised for 20 epoch
        #     train_decoder_local = True
        # else:
        print(f"Train supervised (freeze decoder).")
        
        train_decoder_local = False
        
        for param in model.dec.parameters():
            param.requires_grad = False

        # if epoch == 21 and start_flag <= 0:
        #     # reset loss for regression and norm loss
        #     min_loss = float('inf')

        if epoch == 50:
            criterion.set_ts(1000, 1000)

        if epoch == 80:
            criterion.set_ts(100, 100)

        if epoch == 105:
            criterion.set_ts(10, 10)

        if epoch == 130:
            criterion.set_ts(1, 1)

        start_time = time.time()
        print(f"----- Epoch {epoch} start -----")

        data_loader = tqdm(zip(train_dataset, gt_dataset), 
                           total=len(train_dataset), 
                           desc=f"Epoch {epoch}/{num_epochs}")

        for train_data, gt_data in data_loader:
            # ---- [1] 만약 flag < start_flag이면 => skip
            if flag <= start_flag:
                flag += 1
                continue
            # -----------------------------------------

            if gt_data is None or gt_data.size(0) == 0:
                skipped += 1
                flag += 1
                continue
            
            # print("flag: ", flag)
            optimizer.zero_grad()

            # Forward pass
            pred_feat, pred_decoder = model(train_data, train_decoder_local)

            # Compute loss
            loss = criterion(pred_feat=pred_feat,
                             pred_decoder=pred_decoder,
                             input_data=train_data["feat"],
                             gt_data=gt_data,
                             train_decoder=train_decoder_local)

            # Backprop + update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            rec, reg, norm = criterion.get_loss()
            
            rec_total += rec
            reg_total += reg
            norm_total += norm

            iteration_count += 1
            flag += 1

            # 주기적으로 ckpt/TEASER 저장
            if flag % 3000 == 0:
                save_path = f"/home/server01/js_ws/lidar_test/ckpt/TEASER/latest_encoding_logger.pth"
                torch.save({
                    'epoch': epoch,
                    'flag': flag,  # ★ flag 저장
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'min_loss': min_loss
                }, save_path)
                print(f"Model saved at {save_path}, flag: {flag}, skipped: {skipped}")

        # 에폭 끝
        avg_loss = total_loss / max(1, iteration_count)
        avg_reg = reg_total / max(1, iteration_count)
        avg_norm = norm_total / max(1, iteration_count)

        start_flag = -1
        scheduler.step()

        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
        print(f"Min Loss: {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")

        # wandb log (생략 가능)
        wandb.log({
                   "train_loss": avg_loss, "epoch": epoch,
                   "learning_rate": scheduler.get_last_lr()[0],
                   "norm_avg":avg_norm,
                   "reg_avg": avg_reg
                   })

        # best model check
        if avg_loss < min_loss:
            if avg_loss < 0:
                print(f"ERROR: negative loss, skip saving.")
            else:
                min_loss = avg_loss
                save_path = f"/home/server01/js_ws/lidar_test/ckpt/TEASER/best_model_encoding.pth"
                torch.save({
                    'epoch': epoch,
                    'flag': flag,  # ★ flag 저장
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'min_loss': min_loss
                }, save_path)
                print(f"Best model saved at {save_path} with loss: {min_loss:.4f}")

        # Save latest every epoch
        save_path = f"/home/server01/js_ws/lidar_test/ckpt/TEASER/latest_encoding.pth"
        torch.save({
                'epoch': epoch,
                'flag': flag,  # ★ flag 저장
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
        print(f"Model saved at {save_path}, skipped: {skipped}")
        
        # if epoch == 20:
        #     save_path = f"/home/server01/js_ws/lidar_test/ckpt/TEASER/supervised_learned.pth"
        #     torch.save({
        #             'epoch': epoch,
        #             'flag': flag,   # ★ flag 저장
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'scheduler_state_dict': scheduler.state_dict(),
        #             'min_loss': min_loss
        #         }, save_path)
        #     print(f"Model saved at {save_path}, skipped: {skipped}")
        
        if epoch == num_epochs:
            save_path = f"/home/server01/js_ws/lidar_test/ckpt/TEASER/final_results.pth"
            torch.save({
                    'epoch': epoch,
                    'flag': flag,   # ★ flag 저장
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'min_loss': min_loss
                }, save_path)
            print(f"Final Model saved at {save_path}, skipped: {skipped}")
        
        print(f"================================")

    print("========== train complete ==========")
    return

# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--resume-from')
args = parser.parse_args()

device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("training device: ", device_train)
ascii_art = pyfiglet.figlet_format("LIDAR-TEASER-ENCODING")
print(ascii_art)

wandb.init(project="lidar_encoding_TEASER", name="LiDARENCODING-TEASER")

model = PTEncoder(
                 in_channels = 4,
                 drop_path = 0.3,
                 enc_depths = (2, 2, 4, 6, 4, 2, 2),
                 enc_channels = (32, 64, 128, 256, 512, 256, 128),
                 enc_num_head = (2, 4, 8, 16, 32, 16, 8),
                 enc_patch_size = (1024, 1024, 1024, 1024, 1024, 1024, 1024), 
                 qkv_bias = True, 
                 qk_scale = None, 
                 attn_drop = 0.1, 
                 proj_drop = 0.1, 
                 mlp_ratio = 4, 
                 stride = (2, 2, 4, 4, 4, 2),
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 out_channels=6,
                 dec_depths=(1, 1, 2, 2, 1, 1),
                 dec_channels=(4, 8, 16, 32, 64, 64),
                 dec_num_head=(2, 2, 4, 8, 8, 8),
                 dec_patch_size=(1024, 1024, 1024, 1024, 1024, 1024),
)

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 3.0)
train = []; gt = []

train_file_paths = "/home/server01/js_ws/dataset/reconstruction_dataset/reconstruction_input/train/velodyne/"
gt_file_paths = "/home/server01/js_ws/dataset/encoder_dataset_TEASER/encoder_xyzn/"

for seq in range(2,11): # train for seq 02 to seq 11
    train += glob.glob(train_file_paths + f"{seq:02d}/*.bin")
    gt += glob.glob(gt_file_paths + f"{seq:02d}/*.bin")

train_dataset = PointCloudDataset(train, device_train, grid_size=0.01)  # 16channel lidar
gt_dataset = PointCloudGTDataset(gt, device_train)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-9)

scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,3,4,5], gamma=10)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,180], gamma=0.5)

scheduler = ChainedScheduler([scheduler1, scheduler2], optimizer)

criterion = CombinedCriterionAETeaser(device=device_train)

start_flag = 0
start_epoch = 1
num_epochs = 200
min_loss = float('inf')

if args.resume_from:
    ckptr = torch.load(args.resume_from, map_location=device_train)
        
    model.load_state_dict(ckptr['model_state_dict'])
    model.to(device_train)
    # optimizer.load_state_dict(ckptr['optimizer_state_dict'])
    # scheduler.load_state_dict(ckptr['scheduler_state_dict'])
    # min_loss = ckptr['min_loss']

    # start_epoch = ckptr['epoch']
    # start_flag = ckptr['flag']
        
    # if start_flag == 0 or start_flag == len(train_dataset):  #flag == 0
    #     start_epoch += 1
    #     start_flag = 0

        # if start_epoch == 21:

    if start_epoch == num_epochs:
        print("---- Train already finished!! ----")
        sys.exit()

    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    print(f"min loss {min_loss:.4f}")
    print(f"start flag {start_flag}")
    print(f"lr {scheduler.get_last_lr()[0]}")
else:
    start_epoch = 1
    min_loss = float('inf')
    print(f"Starting training from scratch.")

train_encoding(model = model, train_dataset = train_dataset, gt_dataset = gt_dataset, device_train = device_train, scheduler = scheduler, 
               optimizer = optimizer, criterion = criterion, start_epoch=start_epoch, min_loss=min_loss, start_flag = start_flag)