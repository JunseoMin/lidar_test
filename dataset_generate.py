import torch
import numpy as np
from glob import glob
from model.PTv3 import Point
import random
import os
from multiprocessing import Pool

import hashlib

def load_kitti_bin(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def kitti_to_point(file_path, grid_size=0.05, batch_id=0):
    raw_data = load_kitti_bin(file_path)
    coords = raw_data[:, :3]  # x, y, z
    intensity = raw_data[:, 3:4]  # intensity

    features = torch.cat([
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(intensity, dtype=torch.float32)
    ], dim=1)

    batch_tensor = torch.full((features.shape[0],), batch_id, dtype=torch.int64)

    return Point({
        "coord": features[:, :3],
        "feat": features,
        "batch": batch_tensor,
        "grid_size": torch.tensor([grid_size]),
    })


def generate_sparse_point_cloud(points, sparse_factor=1):
    num_points = points.shape[0]
    num_sparse_points = max(1, num_points // sparse_factor)
    indices = random.sample(range(num_points), num_sparse_points)
    return points[indices, :]


def save_point_cloud_to_bin(points, output_file_path):
    points = points.astype(np.float32)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    points.tofile(output_file_path)


def generate_unique_filename(file_path, output_dir):
    """
    Generate a unique filename by hashing the file path.
    :param file_path: Original file path
    :param output_dir: Output directory
    :return: Unique file path
    """
    # 파일 경로 해싱 (중복 방지)
    hash_digest = hashlib.md5(file_path.encode('utf-8')).hexdigest()
    base_name = os.path.basename(file_path)
    unique_name = f"{os.path.splitext(base_name)[0]}_{hash_digest[:8]}.bin"
    return os.path.join(output_dir, unique_name)


def process_file(file_path, train_dir, gt_dir, sparse_factor, grid_size=0.05, batch_id=0):
    point = kitti_to_point(file_path, grid_size=grid_size, batch_id=batch_id)
    raw_points = point["feat"].numpy()

    gt_output_path = generate_unique_filename(file_path, gt_dir)
    save_point_cloud_to_bin(raw_points, gt_output_path)

    sparse_points = generate_sparse_point_cloud(raw_points, sparse_factor=sparse_factor)
    train_output_path = generate_unique_filename(file_path, train_dir)
    save_point_cloud_to_bin(sparse_points, train_output_path)

    print(f"Processed: {file_path} -> GT: {gt_output_path}, Sparse: {train_output_path}")


def process_files_in_parallel(file_paths, train_dir, gt_dir, sparse_factor, num_workers=4):
    args = [(fp, train_dir, gt_dir, sparse_factor) for fp in file_paths]
    with Pool(num_workers) as pool:
        pool.starmap(process_file, args)


file_paths = glob("/home/server01/js_ws/dataset/**/*.bin", recursive=True)
train_ratio = 0.9
num_train = int(len(file_paths) * train_ratio)
random.shuffle(file_paths)

train_files = file_paths[:num_train]
test_files = file_paths[num_train:]

train_output_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti/train"
test_output_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti/test"
gt_output_dir = "/home/server01/js_ws/dataset/sparse_pointclouds_kitti/GT"

sparse_factor = 32

process_files_in_parallel(train_files, train_output_dir, gt_output_dir, sparse_factor, num_workers=20)
process_files_in_parallel(test_files, test_output_dir, gt_output_dir, sparse_factor, num_workers=20)
