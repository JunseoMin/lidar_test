import torch
from model.LidarUpsample import Lidar4US, Point
import numpy as np
import os
import glob
import time
import sys

def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def kitti_to_dict(file_path, device, grid_size=0.05, segments=3):
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
    # batch_tensor = torch.full((features.shape[0],), 0, dtype=torch.int64)

    # print(batch_tensor)
    return {
        "coord": features[:, :3],
        "feat": features,
        "batch": batch_tensor,
        "grid_size": torch.tensor(grid_size, device=device)
    }

def point_to_bin(output, output_path):
    output = output.cpu().numpy()
    output.tofile(output_path)

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
            # print(point_dict.keys())
            # print(point_dict['feat'].shape)
            
            start_time = time.time()
            output = model(point_dict)
            # print(output.shape)
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
    print(f"Using device: {device}")
    print("current device: ", torch.cuda.current_device())
    print("count of GPUs: ", torch.cuda.device_count())
    
    if torch.cuda.device_count() != 1:
        sys.exit("Only one GPU is supported")

    
    ckpt_dir = "/home/server01/js_ws/lidar_test/ckpt/best_model.pth"
    input_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti/test/0.bin"
    output_dir = "/home/server01/js_ws/lidar_test/evaluate_output"
    
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
    )
    
    checkpoint = torch.load(ckpt_dir, map_location="cuda", weights_only=True)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    print("model set!")
    # try:
    avg_inference_time = evaluate_model(model, input_dir, output_dir, device)
    print(f"Evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Average inference time: {avg_inference_time:.4f}s")
    # except Exception as e:
    #     print(f"Evaluation failed with error: {str(e)}")

if __name__ == '__main__':
    # try:
    main()
    # except Exception as e:
    #     pass
    #     print(f"Exception in main: {str(e)}")
