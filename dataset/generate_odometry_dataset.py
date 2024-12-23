import torch
import numpy as np
from glob import glob
import random
import os
from multiprocessing import Pool
from sklearn.cluster import DBSCAN

def load_kitti_bin(file_path):
    """
    Load point cloud data from a .bin file (KITTI format).
    Each point is [x, y, z, intensity].
    """
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def load_kitti_label(file_path):
    """
    Load label data from a .label file (Semantic KITTI format).
    Returns:
        semantic_label: np.ndarray shape (N,) semantic class labels
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label file not found: {file_path}")
    label_data = np.fromfile(file_path, dtype=np.uint32)
    # Extract semantic label (lower 16 bits)
    semantic_label = label_data & 0xFFFF
    return semantic_label

def generate_sparse_point_cloud(points, sparse_factor=4, vertical_angle_bins=64, min_angle=-25.0, max_angle=2.0):
    """
    Reduce 64-channel point cloud (HDL-64E) to 16-channel by selecting points in every 'sparse_factor'-th vertical angle bin.
    This is done by first computing the vertical angle of each point and grouping them into fixed, known-angle bins.

    Args:
        points (np.ndarray): shape (N,4) point cloud.
        sparse_factor (int): factor by which we downsample the channels. For 64->16, set to 4.
        vertical_angle_bins (int): total vertical bins (64 for HDL-64)
        min_angle (float): minimum vertical angle (e.g. -25.0 deg)
        max_angle (float): maximum vertical angle (e.g. 2.0 deg)

    Returns:
        np.ndarray: Downsampled point cloud of shape (M,4)
    """
    distances = np.linalg.norm(points[:, :3], axis=1)
    distances[distances == 0] = 1e-6
    vertical_angles = np.degrees(np.arcsin(points[:, 2] / distances))

    bin_edges = np.linspace(min_angle, max_angle, vertical_angle_bins + 1)
    bin_indices = np.digitize(vertical_angles, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, vertical_angle_bins - 1)

    selected_points = []
    for i in range(0, vertical_angle_bins, sparse_factor):
        mask = (bin_indices == i)
        if np.any(mask):
            selected_points.append(points[mask])

    if selected_points:
        selected_points = np.vstack(selected_points)
    else:
        selected_points = np.empty((0, 4), dtype=np.float32)

    return selected_points

def save_point_cloud_to_bin(points, output_file_path):
    """
    Save point cloud data to a .bin file.
    points can be Nx4 or Nx5 etc. Just ensure float32.
    """
    points = points.astype(np.float32)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    points.tofile(output_file_path)

def generate_unique_filename(file_path, output_dir, idx):
    """
    Generate a unique filename for output files based on index and output directory.
    """
    unique_name = f"{idx:06d}.bin"
    return os.path.join(output_dir, unique_name)

def process_file(file_path, label_paths, input_dir, gt_dir, sparse_factor, idx):
    """
    Process a single .bin file:
    - Load raw points (64-channel)
    - Load corresponding label
    - Save the original (GT) point cloud with label: [x, y, z, intensity, label]
    - Generate and save the sparse (16-channel) point cloud ([x, y, z, intensity] only)
    """
    # Load raw points
    raw_points = load_kitti_bin(file_path)

    # Derive label file path from the bin file path
    label_file_path = label_paths
    semantic_label = load_kitti_label(label_file_path)  # shape (N,)
    if semantic_label.shape[0] != raw_points.shape[0]:
        raise ValueError(f"Mismatch between points and label length: {raw_points.shape[0]} vs {semantic_label.shape[0]}")

    # Save GT file (original) with label
    # Combine raw points and semantic label to [x, y, z, intensity, label]
    gt_points = np.hstack((raw_points, semantic_label[:, None].astype(np.float32)))
    gt_output_path = generate_unique_filename(file_path, gt_dir, idx)
    save_point_cloud_to_bin(gt_points, gt_output_path)

    # Generate sparse (16-channel) point cloud (no label in this one, only for training)
    sparse_points = generate_sparse_point_cloud(raw_points, sparse_factor=sparse_factor)
    train_output_path = generate_unique_filename(file_path, input_dir, idx)
    save_point_cloud_to_bin(sparse_points, train_output_path)

    print(f"Processed: {file_path} -> GT: {gt_output_path} (with label), Sparse: {train_output_path}")

def process(dataset_path_pattern, label_path, train_dir, gt_dir, sparse_factor):
    """
    Process all .bin files matched by dataset_path_pattern into train_dir and gt_dir.
    """
    file_paths = glob(dataset_path_pattern)
    label_paths = glob(label_path)
    file_paths.sort()
    label_paths.sort()
    # random.shuffle(file_paths)

    for i, f in enumerate(file_paths):
        print("current processing: ",f)
        print("current processing: ",label_paths[i])
        process_file(file_path=f,label_paths = label_paths[i], input_dir=train_dir, gt_dir= gt_dir, sparse_factor=sparse_factor, idx=i)

    return len(file_paths)


# main
sparse_factor = 4  # 64 channels -> 16 channels
for i in range():
    dataset_path = f"/home/server01/js_ws/dataset/odometry_dataset/dataset/sequences/{i:02d}/velodyne/*.bin"
    label_path = f"/home/server01/js_ws/dataset/odometry_dataset/dataset/sequences/{i:02d}/labels/*.label"
    # 동일 이름의 .label 파일이 존재한다고 가정
    # 예: 000000.bin -> 000000.label
    
    train_output_dir = f"/home/server01/js_ws/dataset/odometry_dataset/train/{i:02d}"
    gt_output_dir = f"/home/server01/js_ws/dataset/odometry_dataset/gt/{i:02d}"

    dataset_len = process(dataset_path, label_path, train_output_dir, gt_output_dir, sparse_factor)
    print(f"Sequence {i:02d} finished! Total processed frames: {dataset_len}")
