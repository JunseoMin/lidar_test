# CUDA_VISIBLE_DEVICES=1 python3 evaluate_overfit.py
import torch
from model.LidarUpsample import Lidar4US, Point
import numpy as np
import os
import glob
import time
import sys

def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def kitti_to_dict(file_path, device, grid_size=0.05, segments=1):
    raw_data = load_kitti_bin(file_path)
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity as a feature

    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32, device=device),
        torch.tensor(intensity, dtype=torch.float32, device=device)
    ], dim=1)

    num_points = features.shape[0]
    segments = segments

    points_per_segment = (num_points + segments - 1) // segments

    batch_tensor = torch.arange(segments, device=device).repeat_interleave(points_per_segment)[:num_points]
    batch_tensor = batch_tensor.to(dtype=torch.int64)

    return {
        "coord": features[:, :3],
        "feat": features,
        "batch": batch_tensor,
        "grid_size": torch.tensor(grid_size, device=device)
    }

def point_to_bin(output, output_path):
    # Convert to numpy and ensure it's on CPU
    a = output
    output = output.cpu().numpy()
    # Reshape to (N, 3) format for [x,y,z]
    points = output.reshape(-1, 3)
    # Save to binary file
    points.astype(np.float32).tofile(output_path)

def read_kitti_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32)
    points = points.reshape(-1, 4)  # Reshape to (N, 4) - KITTI format
    return points

def evaluate_model(model, input_dir, output_dir, device="cuda:1"):
    model.to(device)
    model.eval()
    total_time = 0
    total_files = 0
    
    test_files = glob.glob(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for test_file in test_files:
            filename = os.path.basename(test_file)
            point_dict = kitti_to_dict(test_file, device=device)
            
            start_time = time.time()
            output = model(point_dict)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_files += 1
            
            output_path = os.path.join(output_dir, filename)
            point_to_bin(output.feat, output_path)
            
            print(f"Processed {filename}, inference time: {end_time - start_time:.4f}s")
    
    avg_time = total_time / total_files
    print(f"\nAverage inference time: {avg_time:.4f}s per file")
    return avg_time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() != 1:
        sys.exit("Only one GPU is supported")

    
    ckpt_dir = "/home/server01/js_ws/lidar_test/ckpt/latest_vertical_upsample.pth"
    # input_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti/test/1000.bin"
    input_dir = "/home/server01/js_ws/dataset/vertical_downsampled/test/100.bin"
    # input_gt_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti/gt/0.bin"
    output_dir = "/home/server01/js_ws/lidar_test/evaluate_output/vertical_upsample"
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
    
    checkpoint = torch.load(ckpt_dir, map_location="cuda", weights_only=True)
    print(checkpoint['epoch'])
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        avg_inference_time = evaluate_model(model, input_dir, output_dir, device)
        print(f"Evaluation completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Average inference time: {avg_inference_time:.4f}s")
    except Exception as e:
        print(f"Evaluation failed with error: {str(e)}")

if __name__ == '__main__':
    main()
