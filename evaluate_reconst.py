import torch
from model.LidarUpsample import Lidar4US, Point
import numpy as np
import os
import glob
import time
import sys
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def load_kitti_bin_gt(file_path):
    # print(file_path)
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)

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
    output = output.cpu().numpy()
    output.tofile(output_path)

# Chamfer Distance 계산
def chamfer_distance(gt_points, pred_points):
    dist_1 = 0
    for i in range(len(pred_points)):
        dists = np.linalg.norm(gt_points - pred_points[i], axis=1)
        dist_1 += np.min(dists)

    dist_2 = 0
    for j in range(len(gt_points)):
        dists = np.linalg.norm(pred_points - gt_points[j], axis=1)
        dist_2 += np.min(dists)

    chamfer_dist = (1 / len(pred_points)) * dist_1 + (1 / len(gt_points)) * dist_2
    return chamfer_dist

# IoU 계산
def intersection_over_union(gt_points, pred_points, voxel_size=0.05):
    gt_voxels = np.floor(gt_points / voxel_size).astype(np.int32)
    pred_voxels = np.floor(pred_points / voxel_size).astype(np.int32)

    gt_voxels_set = set(map(tuple, gt_voxels))
    pred_voxels_set = set(map(tuple, pred_voxels))

    intersection = len(gt_voxels_set & pred_voxels_set)
    union = len(gt_voxels_set | pred_voxels_set)

    return intersection / union if union > 0 else 0.0

# MAE 계산
def mean_absolute_error(gt_points, pred_points):
    # 두 집합 중 작은 크기 선택
    num_points = min(len(gt_points), len(pred_points))

    # 작은 크기에 맞게 샘플링
    sampled_pred = pred_points[:num_points]
    sampled_gt = gt_points[:num_points]

    # 각 점 간의 절대 거리 계산
    absolute_errors = np.linalg.norm(sampled_pred - sampled_gt, axis=1)  # 유클리드 거리
    mae = np.mean(absolute_errors)  # 평균 절대 거리

    return mae

def evaluate_model(model, gt_dir, input_dir, output_dir, device="cuda"):
    model.to(device)
    model.eval()
    total_time = 0
    total_files = 0
    total_chamfer_distance = 0
    total_wasserstein_distance = 0
    total_iou = 0
    total_mae = 0
    
    test_files = glob.glob(input_dir)
    gt_files = glob.glob(gt_dir)

    # 파일 정렬
    test_files.sort()
    gt_files.sort()

    # 파일 이름 비교 및 확인
    test_file_names = [os.path.basename(f) for f in test_files]
    gt_file_names = [os.path.basename(f) for f in gt_files]

    if test_file_names != gt_file_names:
        print("Warning: Input and ground truth files do not match!")
        unmatched_files = set(test_file_names).symmetric_difference(gt_file_names)
        print(f"Unmatched files: {unmatched_files}")
        return  # 파일 이름이 일치하지 않으면 평가 중단
    
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        # tqdm을 사용하여 진행 상태를 확인
        for index in tqdm(range(len(test_files)), desc="Processing files"):  
            filename = os.path.basename(test_files[index])
            point_dict = kitti_to_dict(test_files[index], device=device)
            gt_points = load_kitti_bin_gt(gt_files[index])[:, :3]

            print(gt_points)

            start_time = time.time()
            output = model(point_dict)
            pred_points = output.feat.cpu().numpy()[:, :3]
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_files += 1
            
            # Chamfer Distance 계산
            chamfer_dist = chamfer_distance(gt_points, pred_points)
            
            # IoU 계산
            iou = intersection_over_union(gt_points, pred_points)
            
            # MAE 계산
            mae = mean_absolute_error(gt_points, pred_points)

            total_chamfer_distance += chamfer_dist
            total_iou += iou
            total_mae += mae
            
            output_path = os.path.join(output_dir, filename)
            point_to_bin(output.feat, output_path)

            if(total_files >= 100):
                break

    avg_time = total_time / total_files
    avg_chamfer_distance = total_chamfer_distance / total_files
    avg_iou = total_iou / total_files
    avg_mae = total_mae / total_files
    
    print(f"\nAverage inference time: {avg_time:.4f}s per file")
    print(f"Average Chamfer Distance: {avg_chamfer_distance:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    return avg_time, avg_chamfer_distance, avg_iou, avg_mae


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("current device: ", torch.cuda.current_device())
    print("count of GPUs: ", torch.cuda.device_count())

    ckpt_dir = "/home/server01/js_ws/lidar_test/ckpt/latest_reconst.pth"
    input_dir = "/home/server01/js_ws/dataset/reconstruction_input/train/velodyne/02/002582.bin"
    gt_dir = "/home/server01/js_ws/dataset/odometry_dataset/reconstruction_gt/02/002582.bin"
    output_dir = "/home/server01/js_ws/lidar_test/evaluate_output/reconst_test"
    
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
        dec_depths=(2, 2, 2, 4, 2),
        dec_n_head=(2, 2, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_channels=(32, 64, 128, 256, 512),
        train_decoder=True,
        exp_hidden=64,
        exp_out=32,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        num_1x1s=4,
        out_channel=3,
    )
    
    checkpoint = torch.load(ckpt_dir, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("model set!")
    
    # try:
    avg_inference_time, avg_chamfer_distance, avg_iou, avg_mae = evaluate_model(model, gt_dir, input_dir, output_dir, device)
    print(f"Evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Average inference time: {avg_inference_time:.4f}s")
    print(f"Average Chamfer Distance: {avg_chamfer_distance:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    # except Exception as e:
    #     print(f"Evaluation failed with error: {str(e)}")

if __name__ == '__main__':
    # try:
    main()
    # except Exception as e:
    #     print(f"Exception in main: {str(e)}")
