import torch
from model.LidarUpsample import Lidar4US
import numpy as np
import os
import glob
import time

def bin_to_dict_one(bin_path, grid_size=0.05, device="cuda:1"):
    bin = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    
    num_points = bin.shape[0]
    segments = 3
    points_per_segment = (num_points + segments - 1) // segments
    batch_tensor = torch.arange(segments).repeat_interleave(points_per_segment)[:num_points]
    batch_tensor = batch_tensor.to(dtype=torch.int64)
    
    point_dict = {
        'coord': torch.tensor(bin[:,:3], dtype=torch.float32).to(device),
        'feat': torch.tensor(bin[:,3:], dtype=torch.float32).to(device),
        'batch': batch_tensor.to(device),
        'grid_size': torch.tensor(grid_size).to(device)
    }
    
    return point_dict

def evaluate_model(model, input_dir, output_dir, device="cuda:1"):
    model.eval()
    total_time = 0
    total_files = 0
    
    test_files = glob.glob(os.path.join(input_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for test_file in test_files:
            filename = os.path.basename(test_file)
            point_dict = bin_to_dict_one(test_file, device=device)
            
            start_time = time.time()
            output = model(point_dict)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_files += 1
            
            output_path = os.path.join(output_dir, filename)
            point_to_bin(output, output_path)
            
            print(f"Processed {filename}, inference time: {end_time - start_time:.4f}s")
    
    avg_time = total_time / total_files
    print(f"\nAverage inference time: {avg_time:.4f}s per file")
    return avg_time

def main():
    device = "cuda"
    torch.cuda.set_device(device)

    ckpt_dir = "/home/server01/js_ws/lidar_test/ckpt/best_model.pth"
    input_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti/test/0.bin"
    output_dir = "/home/server01/js_ws/lidar_test/evaluate_output"
    
    model = Lidar4US().to(device)
    checkpoint = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    try:
        avg_inference_time = evaluate_model(model, input_dir, output_dir, device)
        print(f"Evaluation completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Average inference time: {avg_inference_time:.4f}s")
    except Exception as e:
        print(f"Evaluation failed with error: {str(e)}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Exception in main: {str(e)}")
